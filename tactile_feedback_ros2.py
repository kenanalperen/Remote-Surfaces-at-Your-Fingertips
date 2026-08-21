#!/usr/bin/env python3

"""
tactile_feedback.py with ROS2 integration - FINAL VERSION
──────────────────────────────────────────────────────────
CASES (selected via Image dropdown 1-3):
  Case 1 (Image 1) — Pure black/white. Sound ON for black, OFF for white.
  Case 2 (Image 2) — 5-colour image. Per-colour Freq/Wave/Amp. White=silent.
  Case 3 (Image 3) — Greyscale. Amplitude = blackness^gamma (exponential).
                     blackness<0.5  → 200 Hz Triangle
                     blackness>=0.5 → 80 Hz Sine

ROS2 TOPICS:
  /unity_robot/position_command  x/z mapped coords, y=0
  /unity_robot/pixel_position    raw pixel x/y, z=0
  /unity_robot/sound_details     x=amp(0-1), y=freq(Hz), z=wave(0-3)

KEYBOARD: F11=fullscreen  ESC=exit fullscreen
"""

import sys
import os
import glob
import argparse
import numpy as np
from PIL import Image as PilImage

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QStatusBar, QSizePolicy, QFrame,
    QSlider, QComboBox, QCheckBox, QSpinBox
)
from PyQt5.QtCore  import Qt, pyqtSignal, QThread, QSize
from PyQt5.QtGui   import QPixmap, QImage, QFont, QPainter, QColor, QPen, QBrush

import sounddevice as sd
import evdev
from evdev import InputDevice, ecodes

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3


# ───────────────────────────────────────────────────────────────────────────────
#  Device / path helpers
# ───────────────────────────────────────────────────────────────────────────────

def find_ir_device(vendor="1ff7", product="0008"):
    for path in sorted(glob.glob("/sys/class/input/event*/device")):
        try:
            vid = open(f"{path}/id/vendor").read().strip()
            pid = open(f"{path}/id/product").read().strip()
            if vid != vendor or pid != product:
                continue
            prop = int(open(f"{path}/properties").read().strip(), 16)
            if prop & 0x2:
                return f"/dev/input/{path.split('/')[-2]}"
        except (OSError, ValueError):
            continue
    return None


def get_images_folder():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")


def get_image_path(image_num: int) -> str:
    return os.path.join(get_images_folder(), f"image_process_{image_num}.png")


# ───────────────────────────────────────────────────────────────────────────────
#  Homography
# ───────────────────────────────────────────────────────────────────────────────

def compute_homography(src_pts, dst_pts):
    A = []
    for (sx, sy), (dx, dy) in zip(src_pts, dst_pts):
        A += [
            [-sx, -sy, -1,   0,   0,  0, dx*sx, dx*sy, dx],
            [  0,   0,  0, -sx, -sy, -1, dy*sx, dy*sy, dy],
        ]
    _, _, Vt = np.linalg.svd(np.array(A, dtype=float))
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def apply_homography(H, x, y):
    p = H @ np.array([x, y, 1.0])
    return p[0] / p[2], p[1] / p[2]


# ───────────────────────────────────────────────────────────────────────────────
#  Pixel helpers
# ───────────────────────────────────────────────────────────────────────────────

def _clamp(np_image, img_x, img_y):
    h, w = np_image.shape[:2]
    return int(max(0, min(img_x, w-1))), int(max(0, min(img_y, h-1)))


def is_black(np_image, img_x, img_y, threshold=128):
    cx, cy = _clamp(np_image, img_x, img_y)
    r, g, b = np_image[cy, cx]
    return (0.299*r + 0.587*g + 0.114*b) < threshold


def classify_colour(np_image, img_x, img_y):
    cx, cy = _clamp(np_image, img_x, img_y)
    r, g, b = int(np_image[cy, cx, 0]), int(np_image[cy, cx, 1]), int(np_image[cy, cx, 2])
    targets = {
        'white': (255, 255, 255), 'black': (0, 0, 0),
        'red':   (255, 0,   0  ), 'green': (0, 255, 0), 'blue': (0, 0, 255),
    }
    best, best_d = 'white', float('inf')
    for name, (tr, tg, tb) in targets.items():
        d = (r-tr)**2 + (g-tg)**2 + (b-tb)**2
        if d < best_d:
            best_d, best = d, name
    return best


def pixel_blackness(np_image, img_x, img_y):
    cx, cy = _clamp(np_image, img_x, img_y)
    r, g, b = np_image[cy, cx]
    return 1.0 - ((0.299*r + 0.587*g + 0.114*b) / 255.0)


# ───────────────────────────────────────────────────────────────────────────────
#  Sound constants
# ───────────────────────────────────────────────────────────────────────────────

WAVEFORMS    = ["Sine", "Square", "Triangle", "Sawtooth"]
WAVEFORM_IDX = {w: float(i) for i, w in enumerate(WAVEFORMS)}

CASE2_COLOURS  = ['black', 'red', 'green', 'blue']
CASE2_DEFAULTS = {
    'black': {'freq': 120, 'wave': 'Square',   'amp': 1.00},
    'red'  : {'freq':  80, 'wave': 'Sawtooth', 'amp': 1.00},
    'green': {'freq': 160, 'wave': 'Sine',     'amp': 0.50},
    'blue' : {'freq': 200, 'wave': 'Triangle', 'amp': 1.00},
}


# ───────────────────────────────────────────────────────────────────────────────
#  Sound Engine
# ───────────────────────────────────────────────────────────────────────────────

class SoundEngine:
    SAMPLE_RATE       = 44100
    BLOCKSIZE         = 128
    DEFAULT_FREQ      = 120
    DEFAULT_AMPLITUDE = 0.40
    DEFAULT_WAVEFORM  = "Square"

    def __init__(self):
        self.active    = False
        self._last_sound_freq     = 120
        self._last_sound_waveform = 'Square'
        self.freq      = self.DEFAULT_FREQ
        self.amplitude = self.DEFAULT_AMPLITUDE
        self.waveform  = self.DEFAULT_WAVEFORM
        self._phase    = 0.0
        self._stream   = None

    def _callback(self, outdata, frames, time_info, status):
        if not self.active:
            outdata.fill(0); self._phase = 0.0; return
        freq = max(1, self.freq)
        amp  = float(np.clip(self.amplitude, 0.0, 1.0))
        wf   = self.waveform
        dt   = freq / self.SAMPLE_RATE
        t    = (self._phase + np.arange(frames, dtype=np.float64) * dt) % 1.0
        if   wf == "Sine":     buf = amp * np.sin(2.0 * np.pi * t)
        elif wf == "Square":   buf = np.where(t < 0.5, amp, -amp)
        elif wf == "Triangle": buf = amp * (4.0 * np.where(t < 0.5, t, 1.0-t) - 1.0)
        else:                  buf = amp * (2.0 * t - 1.0)
        outdata[:, 0] = buf.astype(np.float32)
        self._phase = (self._phase + frames * dt) % 1.0

    def start(self):
        self._stream = sd.OutputStream(
            samplerate=self.SAMPLE_RATE, channels=1, dtype='float32',
            blocksize=self.BLOCKSIZE, latency='low', callback=self._callback)
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop(); self._stream.close(); self._stream = None


# ───────────────────────────────────────────────────────────────────────────────
#  IR Touch Thread
# ───────────────────────────────────────────────────────────────────────────────

class IRTouchThread(QThread):
    touch_pos  = pyqtSignal(int, int)
    touch_down = pyqtSignal()
    touch_up   = pyqtSignal()
    error      = pyqtSignal(str)

    def __init__(self, device_path, parent=None):
        super().__init__(parent)
        self.device_path = device_path
        self._running    = True

    def run(self):
        try:
            device = InputDevice(self.device_path)
        except Exception as exc:
            self.error.emit(str(exc)); return
        x, y, touching = 0, 0, False
        for event in device.read_loop():
            if not self._running: break
            if event.type == ecodes.EV_ABS:
                if   event.code == ecodes.ABS_MT_POSITION_X: x = event.value
                elif event.code == ecodes.ABS_MT_POSITION_Y: y = event.value
            elif event.type == ecodes.EV_KEY and event.code == ecodes.BTN_TOUCH:
                if event.value == 1: touching = True;  self.touch_down.emit()
                else:                touching = False; self.touch_up.emit()
            elif event.type == ecodes.EV_SYN and touching:
                self.touch_pos.emit(x, y)

    def stop(self):
        self._running = False; self.quit(); self.wait(1000)


# ───────────────────────────────────────────────────────────────────────────────
#  TrackerWidget  — image size/position logic untouched
# ───────────────────────────────────────────────────────────────────────────────

_CAL_LABELS = ["1\nTL", "2\nTR", "3\nBL", "4\nBR"]


class TrackerWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #111118; border: 1px solid #333;")
        self._pil_image = None
        self._np_image  = None
        self._display_w = 0
        self._display_h = 0
        self._offset_x  = 0
        self._offset_y  = 0
        self.phase    = 'cal'
        self.cal_step = 0
        self.cursor_x = -1
        self.cursor_y = -1

    def sizeHint(self):        return QSize(100, 100)
    def minimumSizeHint(self): return QSize(100, 100)

    def load_image(self, path):
        try:
            img = PilImage.open(path).convert("RGB")
            self._pil_image = img
            self._np_image  = np.array(img)
            self._rebuild_pixmap()
            return True
        except Exception as exc:
            print(f"[Image] Could not load '{path}': {exc}"); return False

    def _cal_dot_widget(self, idx):
        if self._display_w == 0 or self._pil_image is None: return 0, 0
        iw, ih = self._pil_image.size
        corners = [(0,0),(iw-1,0),(0,ih-1),(iw-1,ih-1)]
        ix, iy = corners[idx]
        return (self._offset_x + int(ix*self._display_w/iw),
                self._offset_y + int(iy*self._display_h/ih))

    def image_to_widget(self, img_x, img_y):
        if self._pil_image is None or self._display_w == 0: return -1, -1
        iw, ih = self._pil_image.size
        return (self._offset_x + int(img_x*self._display_w/iw),
                self._offset_y + int(img_y*self._display_h/ih))

    def resizeEvent(self, event):
        super().resizeEvent(event); self._rebuild_pixmap()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._pil_image is None: return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self.phase == 'cal':
            self._paint_calibration(painter)
        elif self.phase == 'track' and self.cursor_x >= 0:
            self._paint_cursor(painter)
        painter.end()

    def _paint_calibration(self, painter):
        for i in range(4):
            wx, wy = self._cal_dot_widget(i)
            if   i < self.cal_step:  color, R = QColor(30, 210, 100), 14
            elif i == self.cal_step: color, R = QColor(255, 45, 45),  22
            else:                    color, R = QColor(160, 50, 50),  14
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(255,255,255), 2))
            painter.drawEllipse(wx-R, wy-R, 2*R, 2*R)
            painter.setFont(QFont("Monospace", 8, QFont.Bold))
            painter.setPen(QPen(Qt.white))
            painter.drawText(wx-R, wy-R, 2*R, 2*R, Qt.AlignCenter, _CAL_LABELS[i])

    def _paint_cursor(self, painter):
        wx, wy = self.image_to_widget(self.cursor_x, self.cursor_y)
        if wx < 0: return
        R = 7
        painter.setBrush(QBrush(QColor(255, 30, 30)))
        painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
        painter.drawEllipse(wx-R, wy-R, 2*R, 2*R)
        painter.setPen(QPen(QColor(255, 80, 80, 160), 1))
        painter.drawLine(wx-14, wy, wx-R-2, wy)
        painter.drawLine(wx+R+2, wy, wx+14, wy)
        painter.drawLine(wx, wy-14, wx, wy-R-2)
        painter.drawLine(wx, wy+R+2, wx, wy+14)

    def _rebuild_pixmap(self):
        if self._pil_image is None: return
        wl, hl = self.width(), self.height()
        if wl <= 0 or hl <= 0: return
        iw, ih = self._pil_image.size
        scale  = min(wl/iw, hl/ih)
        new_w, new_h = int(iw*scale), int(ih*scale)
        self._offset_x  = (wl - new_w) // 2
        self._offset_y  = (hl - new_h) // 2
        self._display_w = new_w
        self._display_h = new_h
        resized = self._pil_image.resize((new_w, new_h), PilImage.LANCZOS)
        data    = resized.tobytes("raw", "RGB")
        qimg    = QImage(data, new_w, new_h, 3*new_w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))


# ───────────────────────────────────────────────────────────────────────────────
#  Main Window
# ───────────────────────────────────────────────────────────────────────────────

_CAL_STEP_HINTS = [
    "Touch dot  1 - Top-Left corner",
    "Touch dot  2 - Top-Right corner",
    "Touch dot  3 - Bottom-Left corner",
    "Touch dot  4 - Bottom-Right corner",
]

_CTRL_SS = """
    QLabel      { color: #aaaacc; background: transparent; font-size: 11px; }
    QSpinBox    { background: #1e2240; color: #e0e0ff; border: 1px solid #334;
                  border-radius: 3px; padding: 1px 4px; min-width: 54px; }
    QSpinBox::up-button, QSpinBox::down-button { width: 16px; }
    QComboBox   { background: #1e2240; color: #e0e0ff; border: 1px solid #334;
                  border-radius: 3px; padding: 1px 6px; min-width: 90px; }
    QComboBox QAbstractItemView { background: #1e2240; color: #e0e0ff;
                                   selection-background-color: #3355aa; }
    QSlider::groove:horizontal { height: 4px; background: #334466; border-radius: 2px; }
    QSlider::handle:horizontal { width: 14px; height: 14px; margin: -5px 0;
                                  background: #5577dd; border-radius: 7px; }
    QSlider::sub-page:horizontal { background: #5577dd; border-radius: 2px; }
    QCheckBox   { color: #e0e0ff; font-size: 11px; }
    QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #556;
                           border-radius: 3px; background: #1e2240; }
    QCheckBox::indicator:checked { background: #5577dd; }
"""


class MainWindow(QMainWindow):

    def __init__(self, device_path, ros_node):
        super().__init__()
        self.ros_node = ros_node
        self.setWindowTitle("Tactile Feedback - Frame-to-Sound Tracker (ROS2)")
        self.resize(1280, 760)
        self.setMinimumSize(800, 600)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Controls row ──────────────────────────────────────────────────────
        ctrl = QFrame()
        ctrl.setStyleSheet(
            "background-color: #0f1628; border-radius: 6px; border: 1px solid #223;" + _CTRL_SS)
        ctrl_row = QHBoxLayout(ctrl)
        ctrl_row.setContentsMargins(14, 6, 14, 6)
        ctrl_row.setSpacing(14)

        # Freq
        ctrl_row.addWidget(QLabel("Freq (Hz):"))
        self.spin_freq = QSpinBox()
        self.spin_freq.setRange(20, 2000)
        self.spin_freq.setValue(SoundEngine.DEFAULT_FREQ)
        self.spin_freq.setSingleStep(10)
        ctrl_row.addWidget(self.spin_freq)

        # Wave
        ctrl_row.addWidget(QLabel("Wave:"))
        self.combo_wave = QComboBox()
        for wf in WAVEFORMS:
            self.combo_wave.addItem(wf)
        self.combo_wave.setCurrentText(SoundEngine.DEFAULT_WAVEFORM)
        ctrl_row.addWidget(self.combo_wave)

        # Amp
        ctrl_row.addWidget(QLabel("Amp:"))
        self.slider_amp = QSlider(Qt.Horizontal)
        self.slider_amp.setRange(0, 100)
        self.slider_amp.setValue(int(SoundEngine.DEFAULT_AMPLITUDE * 100))
        self.slider_amp.setFixedWidth(110)
        ctrl_row.addWidget(self.slider_amp)
        self.lbl_amp_val = QLabel(f"{int(SoundEngine.DEFAULT_AMPLITUDE * 100)} %")
        self.lbl_amp_val.setFixedWidth(38)
        ctrl_row.addWidget(self.lbl_amp_val)

        # Image selector (1-3)
        ctrl_row.addWidget(QLabel("Image:"))
        self.combo_image = QComboBox()
        for i in range(1, 4):
            self.combo_image.addItem(str(i), i)
        self.combo_image.setCurrentIndex(0)
        ctrl_row.addWidget(self.combo_image)

        # Case-1: Reverse checkbox
        self.chk_reverse = QCheckBox("Reverse")
        self.chk_reverse.setChecked(False)
        self.chk_reverse.setToolTip("Case 1: swap black/white trigger")
        ctrl_row.addWidget(self.chk_reverse)

        # Case-2: colour picker (hidden by default)
        self.lbl_col_pick = QLabel("Colour:")
        self.combo_colour = QComboBox()
        for col in CASE2_COLOURS:
            self.combo_colour.addItem(col.capitalize(), col)
        self.lbl_col_pick.setVisible(False)
        self.combo_colour .setVisible(False)
        ctrl_row.addWidget(self.lbl_col_pick)
        ctrl_row.addWidget(self.combo_colour)

        # Case-3: gamma slider (hidden by default)
        self.lbl_gamma_lbl = QLabel("Curve γ:")
        self.slider_gamma  = QSlider(Qt.Horizontal)
        self.slider_gamma.setRange(10, 50)   # maps to 1.0 … 5.0
        self.slider_gamma.setValue(20)        # default γ=2.0
        self.slider_gamma.setFixedWidth(90)
        self.slider_gamma.setToolTip(
            "Amplitude curve exponent for Case 3.\n"
            "γ=1 linear  γ=2 → at 50% grey amp=25%  γ=3 even steeper")
        self.lbl_gamma_val = QLabel("γ=2.0")
        self.lbl_gamma_val.setFixedWidth(42)
        self.lbl_gamma_lbl.setVisible(False)
        self.slider_gamma  .setVisible(False)
        self.lbl_gamma_val .setVisible(False)
        ctrl_row.addWidget(self.lbl_gamma_lbl)
        ctrl_row.addWidget(self.slider_gamma)
        ctrl_row.addWidget(self.lbl_gamma_val)

        ctrl_row.addStretch()
        root.addWidget(ctrl)

        # ── Info bar ──────────────────────────────────────────────────────────
        info = QFrame()
        info.setStyleSheet("background-color: #16213e; border-radius: 6px;")
        info_row = QHBoxLayout(info)
        info_row.setContentsMargins(12, 6, 12, 6)
        info_row.setSpacing(20)
        font_mono = QFont("Monospace", 11)
        font_mono.setBold(True)
        self.lbl_step   = self._lbl(_CAL_STEP_HINTS[0], font_mono, "#ffaa00")
        self.lbl_colour = self._lbl("--",               font_mono, "#e0e0e0")
        self.lbl_pos    = self._lbl("Pos: --",          font_mono, "#888888")
        self.lbl_sound  = self._lbl("Mute",             font_mono, "#555555")
        self.lbl_device = self._lbl(f"IR: {device_path}", font_mono, "#4466aa")
        self.lbl_ros2   = self._lbl("ROS2: --",         font_mono, "#44aa44")
        for w in (self.lbl_step, self.lbl_colour, self.lbl_pos,
                  self.lbl_sound, self.lbl_device, self.lbl_ros2):
            info_row.addWidget(w)
        info_row.addStretch()
        root.addWidget(info)

        # ── TrackerWidget (stretch=1, owns all remaining vertical space) ──────
        self.tracker = TrackerWidget()
        root.addWidget(self.tracker, stretch=1)

        # ── Wire signals ──────────────────────────────────────────────────────
        self.spin_freq   .valueChanged      .connect(self._on_freq_changed)
        self.combo_wave  .currentTextChanged.connect(self._on_wave_changed)
        self.slider_amp  .valueChanged      .connect(self._on_amp_changed)
        self.combo_image .currentIndexChanged.connect(self._on_image_changed)
        self.combo_colour.currentIndexChanged.connect(self._on_colour_sel_changed)
        self.slider_gamma.valueChanged      .connect(self._on_gamma_changed)

        self.setStatusBar(QStatusBar())
        self._dark_theme()

        # ── State ─────────────────────────────────────────────────────────────
        self._current_image_num = 1
        self._cal_ir_pts  = []
        self._cal_samples = []
        self._collecting  = False
        self._H           = None

        self._sound       = SoundEngine()
        self._sound.start()
        self._was_active  = False
        self._last_colour = None

        # Case-2: per-colour settings (deep copy of defaults)
        self._c2 = {col: dict(CASE2_DEFAULTS[col]) for col in CASE2_COLOURS}

        # Case-3 gamma
        self._c3_gamma = 2.0

        # Position mapping
        self.TACTILE_WIDTH      = 1920
        self.TACTILE_HEIGHT_MAX = 900

        # IR thread
        self._ir = IRTouchThread(device_path)
        self._ir.touch_pos .connect(self._on_pos)
        self._ir.touch_down.connect(self._on_down)
        self._ir.touch_up  .connect(self._on_up)
        self._ir.error     .connect(self._on_ir_error)
        self._ir.start()

        if not self._load_current_image():
            self.statusBar().showMessage("Could not load initial image from images folder")
        else:
            self.statusBar().showMessage(
                "CALIBRATION - touch the red dots on the frame in numbered order (1→2→3→4)")

    # ── Control slots ──────────────────────────────────────────────────────────

    def _on_freq_changed(self, value):
        case = self._current_image_num
        if case == 1:
            self._sound.freq = value
        elif case == 2:
            col = self.combo_colour.currentData()
            self._c2[col]['freq'] = value
            if col == self._last_colour:
                self._sound.freq = value
        # case 3: freq driven by blackness threshold, not slider

    def _on_wave_changed(self, text):
        case = self._current_image_num
        if case == 1:
            self._sound.waveform = text
        elif case == 2:
            col = self.combo_colour.currentData()
            self._c2[col]['wave'] = text
            if col == self._last_colour:
                self._sound.waveform = text
        # case 3: waveform driven by blackness threshold

    def _on_amp_changed(self, value):
        case = self._current_image_num
        frac = value / 100.0
        self.lbl_amp_val.setText(f"{value} %")
        if case == 1:
            self._sound.amplitude = frac
        elif case == 2:
            col = self.combo_colour.currentData()
            self._c2[col]['amp'] = frac
            if col == self._last_colour:
                self._sound.amplitude = frac
        # case 3: read-only display

    def _on_gamma_changed(self, value):
        self._c3_gamma = value / 10.0
        self.lbl_gamma_val.setText(f"γ={self._c3_gamma:.1f}")

    def _on_image_changed(self, index):
        self._current_image_num = int(self.combo_image.itemData(index))
        case = self._current_image_num

        # Show/hide case-specific widgets
        self.chk_reverse  .setVisible(case == 1)
        self.lbl_col_pick .setVisible(case == 2)
        self.combo_colour .setVisible(case == 2)
        self.lbl_gamma_lbl.setVisible(case == 3)
        self.slider_gamma .setVisible(case == 3)
        self.lbl_gamma_val.setVisible(case == 3)

        # Amp slider read-only in case 3; freq/wave locked in case 3
        self.slider_amp.setEnabled(case != 3)
        self.spin_freq .setEnabled(case != 3)
        self.combo_wave.setEnabled(case != 3)

        self._last_colour  = None
        self._sound.active = False

        if not self._load_current_image():
            self.statusBar().showMessage(
                f"Could not load image_process_{self._current_image_num}.png")

        self._refresh_controls_for_case()

    def _on_colour_sel_changed(self, _index):
        self._refresh_controls_for_case()

    def _refresh_controls_for_case(self):
        case = self._current_image_num
        for w in (self.spin_freq, self.combo_wave, self.slider_amp):
            w.blockSignals(True)
        if case == 1:
            self.spin_freq .setValue(self._sound.freq)
            self.combo_wave.setCurrentText(self._sound.waveform)
            self.slider_amp.setValue(int(self._sound.amplitude * 100))
            self.lbl_amp_val.setText(f"{int(self._sound.amplitude * 100)} %")
        elif case == 2:
            col = self.combo_colour.currentData()
            s   = self._c2[col]
            self.spin_freq .setValue(s['freq'])
            self.combo_wave.setCurrentText(s['wave'])
            self.slider_amp.setValue(int(s['amp'] * 100))
            self.lbl_amp_val.setText(f"{int(s['amp'] * 100)} %")
        elif case == 3:
            self.slider_gamma.setValue(int(self._c3_gamma * 10))
            self.lbl_gamma_val.setText(f"γ={self._c3_gamma:.1f}")
            self.slider_amp.setValue(0)
            self.lbl_amp_val.setText("0 %")
        for w in (self.spin_freq, self.combo_wave, self.slider_amp):
            w.blockSignals(False)

    # ── IR slots ───────────────────────────────────────────────────────────────

    def _on_ir_error(self, msg):
        self.statusBar().showMessage(f"IR device error: {msg}")
        self.lbl_device.setText("Device not found")
        self.lbl_device.setStyleSheet("color: #ff4444; background: transparent;")

    def _on_down(self):
        if self.tracker.phase == 'cal':
            self._cal_samples = []; self._collecting = True

    def _on_up(self):
        if self.tracker.phase == 'cal':
            if self._collecting and self._cal_samples:
                self._collecting = False
                arr = np.array(self._cal_samples, dtype=float)
                self._accept_cal_point(int(np.median(arr[:, 0])),
                                       int(np.median(arr[:, 1])))
        else:
            self._set_sound(False)
            self.ros_node.publish_sound_details(0.0, self._last_sound_freq, self._last_sound_waveform)
            self.tracker.cursor_x = -1
            self.tracker.cursor_y = -1
            self.tracker.update()
            self.lbl_colour.setText("--")
            self.lbl_colour.setStyleSheet("color: #e0e0e0; background: transparent;")
            self.lbl_pos.setText("Pos: --")
            self.lbl_ros2.setText("ROS2: --")

    def _on_pos(self, x, y):
        if self.tracker.phase == 'cal':
            if self._collecting: self._cal_samples.append((x, y))
        else:
            self._track(x, y)

    # ── Calibration ────────────────────────────────────────────────────────────

    def _accept_cal_point(self, ir_x, ir_y):
        step = self.tracker.cal_step
        self._cal_ir_pts.append((ir_x, ir_y))
        self.tracker.cal_step += 1
        self.tracker.update()
        self.statusBar().showMessage(f"Dot {step+1} recorded at IR ({ir_x}, {ir_y})")
        if self.tracker.cal_step == 4:
            self._finish_calibration()
        else:
            self.lbl_step.setText(_CAL_STEP_HINTS[self.tracker.cal_step])

    def _finish_calibration(self):
        iw, ih = self.tracker._pil_image.size
        img_corners = [(0,0),(iw-1,0),(0,ih-1),(iw-1,ih-1)]
        self._H = compute_homography(self._cal_ir_pts, img_corners)
        self.tracker.phase = 'track'
        self.lbl_step.setText("Calibrated - tracking")
        self.lbl_step.setStyleSheet("color: #00ff88; background: transparent;")
        self.statusBar().showMessage("Calibration complete! Touch the frame to track.")

    # ── Tracking dispatcher ────────────────────────────────────────────────────

    def _track(self, ir_x, ir_y):
        if self._H is None: return
        fx, fy = apply_homography(self._H, ir_x, ir_y)
        img_x, img_y = int(fx), int(fy)

        self.tracker.cursor_x = img_x
        self.tracker.cursor_y = img_y
        self.tracker.update()

        case = self._current_image_num
        if   case == 1: self._track_case1(img_x, img_y)
        elif case == 2: self._track_case2(img_x, img_y)
        elif case == 3: self._track_case3(img_x, img_y)

        self.lbl_pos.setText(f"Pos: ({img_x:4d}, {img_y:4d})")

        x_ros2, y_ros2, z_ros2 = self._map_position_to_ros2(img_x, img_y)
        self.ros_node.publish_position(x_ros2, y_ros2, z_ros2)
        self.ros_node.publish_pixel_position(img_x, img_y)
        # Always update last known freq/waveform when sound is active
        if self._sound.active:
            self._last_sound_freq     = self._sound.freq
            self._last_sound_waveform = self._sound.waveform
        # Always publish — amplitude 0 when silent, last known freq/wave retained
        self.ros_node.publish_sound_details(
            self._sound.amplitude if self._sound.active else 0.0,
            self._last_sound_freq,
            self._last_sound_waveform)
        self.lbl_ros2.setText(f"ROS2: x={x_ros2:.3f}, z={z_ros2:.3f}")

    # ── Case handlers ──────────────────────────────────────────────────────────

    def _track_case1(self, img_x, img_y):
        """Case 1: pure black/white."""
        black    = is_black(self.tracker._np_image, img_x, img_y)
        reverse  = self.chk_reverse.isChecked()
        sound_on = black if not reverse else (not black)
        if sound_on != self._was_active:
            self._set_sound(sound_on)
            self._was_active = sound_on
        self.lbl_colour.setText("BLACK" if black else "WHITE")
        self.lbl_colour.setStyleSheet(
            f"color: {'#cccccc' if black else '#ffffff'}; background: transparent;")

    def _track_case2(self, img_x, img_y):
        """Case 2: 5-colour image with per-colour sound settings."""
        col = classify_colour(self.tracker._np_image, img_x, img_y)
        col_hex = {'white':'#ffffff','black':'#cccccc',
                   'red':'#ff4444','green':'#44ff44','blue':'#4488ff'}
        self.lbl_colour.setText(col.upper())
        self.lbl_colour.setStyleSheet(
            f"color: {col_hex.get(col,'#ffffff')}; background: transparent;")

        if col == 'white':
            self._set_sound(False)
            self._was_active  = False
            self._last_colour = col
            return

        if col != self._last_colour:
            s = self._c2[col]
            self._sound.freq      = s['freq']
            self._sound.waveform  = s['wave']
            self._sound.amplitude = s['amp']
            self._last_colour = col

        self._set_sound(True)
        self._was_active = True

    def _track_case3(self, img_x, img_y):
        """
        Case 3: greyscale with exponential amplitude mapping.
          amplitude = blackness ^ gamma
          blackness < 0.5  → 200 Hz Triangle
          blackness >= 0.5 → 80 Hz Sine
        """
        blackness = pixel_blackness(self.tracker._np_image, img_x, img_y)
        gamma     = self._c3_gamma

        # Exponential amplitude: slower rise at low blackness, faster near black
        amplitude = blackness ** gamma

        # Frequency and waveform depend on blackness threshold
        if blackness < 0.5:
            self._sound.freq     = 200
            self._sound.waveform = 'Triangle'
        else:
            self._sound.freq     = 80
            self._sound.waveform = 'Sine'

        self._sound.amplitude = amplitude
        sound_on = amplitude > 0.01
        self._set_sound(sound_on)
        self._was_active = sound_on

        # Update amp display (read-only for case 3)
        pct = int(amplitude * 100)
        self.lbl_amp_val.setText(f"{pct} %")
        self.slider_amp.blockSignals(True)
        self.slider_amp.setValue(pct)
        self.slider_amp.blockSignals(False)

        luma_hex = int((1.0 - blackness) * 200) + 55
        self.lbl_colour.setText(f"GREY {int(blackness*100)}%")
        self.lbl_colour.setStyleSheet(
            f"color: rgb({luma_hex},{luma_hex},{luma_hex}); background: transparent;")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _map_position_to_ros2(self, tactile_x, tactile_y):
        cx = min(max(tactile_x, 0), self.TACTILE_WIDTH)
        cy = min(max(tactile_y, 0), self.TACTILE_HEIGHT_MAX)
        return (0.8 - (1.6 * (cx / self.TACTILE_WIDTH)),
                0.0,
                -1.05 + (0.75 * (cy / self.TACTILE_HEIGHT_MAX)))

    def _load_current_image(self):
        image_path = get_image_path(self._current_image_num)
        success = self.tracker.load_image(image_path)
        if success:
            self.statusBar().showMessage(
                f"Loaded image_process_{self._current_image_num}.png")
        return success

    def _set_sound(self, on):
        self._sound.active = on
        if on:
            self.lbl_sound.setText("SOUND ON")
            self.lbl_sound.setStyleSheet(
                "color: #00ff88; font-weight: bold; background: transparent;")
        else:
            self.lbl_sound.setText("Silent")
            self.lbl_sound.setStyleSheet("color: #555555; background: transparent;")

    @staticmethod
    def _lbl(text, font, color):
        lbl = QLabel(text)
        lbl.setFont(font)
        lbl.setStyleSheet(f"color: {color}; background: transparent;")
        return lbl

    def _dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #0c0c16; }
            QWidget     { background-color: #0c0c16; color: #e0e0e0; }
            QStatusBar  { background-color: #16213e; color: #777;
                          font-size: 10px; padding: 2px 8px; }
        """)

    def closeEvent(self, event):
        self._ir.stop(); self._sound.stop(); super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        elif event.key() == Qt.Key_Escape and self.isFullScreen():
            self.showNormal()
        else:
            super().keyPressEvent(event)


# ───────────────────────────────────────────────────────────────────────────────
#  ROS2 Node
# ───────────────────────────────────────────────────────────────────────────────

class TactileFeedbackNode(Node):
    def __init__(self):
        super().__init__('tactile_feedback_node')
        self.robot_pub = self.create_publisher(Vector3, '/unity_robot/position_command', 10)
        self.pixel_pub = self.create_publisher(Vector3, '/unity_robot/pixel_position',   10)
        self.sound_pub = self.create_publisher(Vector3, '/unity_robot/sound_details',    10)
        self.get_logger().info("✅ Tactile Feedback ROS2 node initialized")
        self.get_logger().info("📤 Publishing to /unity_robot/position_command")
        self.get_logger().info("📤 Publishing to /unity_robot/pixel_position")
        self.get_logger().info("📤 Publishing to /unity_robot/sound_details")

    def publish_position(self, x, y, z):
        msg = Vector3(); msg.x = float(x); msg.y = float(y); msg.z = float(z)
        self.robot_pub.publish(msg)

    def publish_pixel_position(self, px_x, px_y):
        msg = Vector3(); msg.x = float(px_x); msg.y = float(px_y); msg.z = 0.0
        self.pixel_pub.publish(msg)

    def publish_sound_details(self, amplitude, frequency, waveform):
        """x=amplitude(0-1), y=frequency(Hz), z=waveform(0=Sine,1=Square,2=Triangle,3=Sawtooth)"""
        msg = Vector3()
        msg.x = float(np.clip(amplitude, 0.0, 1.0))
        msg.y = float(frequency)
        msg.z = float(WAVEFORM_IDX.get(waveform, 0.0))
        self.sound_pub.publish(msg)


# ───────────────────────────────────────────────────────────────────────────────
#  ROS2 background thread
# ───────────────────────────────────────────────────────────────────────────────

class ROS2Thread(QThread):
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node
        self.daemon   = True

    def run(self):
        try:
            rclpy.spin(self.ros_node)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"[ROS2] Error: {e}")


# ───────────────────────────────────────────────────────────────────────────────
#  Entry point
# ───────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="IR Touch Frame -> Tracker -> Sound -> ROS2")
    p.add_argument("--device", default="",
                   help="evdev path e.g. /dev/input/event20  (default: auto-detect)")
    return p.parse_args()


def main(args=None):
    rclpy.init(args=args)
    parsed_args = parse_args()

    if parsed_args.device:
        device_path = parsed_args.device
        print(f"[IR] Using specified device: {device_path}")
    else:
        device_path = find_ir_device()
        if device_path:
            print(f"[IR] Auto-detected device: {device_path}")
        else:
            device_path = "/dev/input/event20"
            print(f"[IR] WARNING: Could not auto-detect IR frame. Falling back to {device_path}")

    ros_node = TactileFeedbackNode()
    app      = QApplication(sys.argv)
    window   = MainWindow(device_path=device_path, ros_node=ros_node)
    window.resize(1280, 760)
    window.show()

    ros2_thread = ROS2Thread(ros_node)
    ros2_thread.start()

    try:
        exit_code = app.exec_()
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
        exit_code = 0
    finally:
        try: ros_node.destroy_node()
        except Exception: pass
        try: rclpy.shutdown()
        except Exception: pass
        print("👋 Goodbye!")
        sys.exit(exit_code)


if __name__ == '__main__':
    main()