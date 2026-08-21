#!/usr/bin/env python3

"""char_exp.py

ROS2 experiment variant for tactile tracking.

Case 2 uses pre‑rendered PNG images (option1/2/3.png) from the images folder.
Sound is a stable 120 Hz square wave with simple on/off hysteresis.
Logging to Path_following.csv and Response_time.csv works reliably.
Now publishes correct position_command coordinates (same mapping as tactile_feedback_ros2.py).
"""

import argparse
import csv
import glob
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
import sounddevice as sd
from evdev import InputDevice, ecodes
from geometry_msgs.msg import Vector3
from rclpy.node import Node
from PyQt5.QtCore import QPointF, QRectF, Qt, QThread, QTimer, QSize, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QBrush, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


LOGICAL_WIDTH = 1920
LOGICAL_HEIGHT = 1080
CASE1_FREQ = 120.0
CASE1_AMPLITUDE = 1.0
CASE1_WAVEFORM = "Square"
WAVEFORM_IDX = {"Sine": 0.0, "Square": 1.0, "Triangle": 2.0, "Sawtooth": 3.0}


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
    """Return the path to the shared 'images' folder (parent directory + /images)."""
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    images_dir = parent_dir / "images"
    print(f"[INFO] get_images_folder -> {images_dir}")
    return str(images_dir)


def compute_homography(src_pts, dst_pts):
    a = []
    for (sx, sy), (dx, dy) in zip(src_pts, dst_pts):
        a += [
            [-sx, -sy, -1, 0, 0, 0, dx * sx, dx * sy, dx],
            [0, 0, 0, -sx, -sy, -1, dy * sx, dy * sy, dy],
        ]
    _, _, vt = np.linalg.svd(np.array(a, dtype=float))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


def apply_homography(h, x, y):
    p = h @ np.array([x, y, 1.0])
    return p[0] / p[2], p[1] / p[2]


def clamp_value(value, low, high):
    return max(low, min(high, value))


def waveform_index(name):
    return WAVEFORM_IDX.get(name, 0.0)


class SoundEngine:
    SAMPLE_RATE = 44100
    BLOCKSIZE = 128

    def __init__(self):
        self.active = False
        self.freq = CASE1_FREQ
        self.amplitude = CASE1_AMPLITUDE
        self.waveform = CASE1_WAVEFORM
        self._phase = 0.0
        self._stream = None

    def _callback(self, outdata, frames, time_info, status):
        if not self.active:
            outdata.fill(0)
            freq = max(1.0, float(self.freq))
            dt = freq / self.SAMPLE_RATE
            self._phase = (self._phase + frames * dt) % 1.0
            return

        freq = max(1.0, float(self.freq))
        amp = float(np.clip(self.amplitude, 0.0, 1.0))
        dt = freq / self.SAMPLE_RATE
        t = (self._phase + np.arange(frames, dtype=np.float64) * dt) % 1.0

        if self.waveform == "Sine":
            buf = amp * np.sin(2.0 * np.pi * t)
        elif self.waveform == "Square":
            buf = np.where(t < 0.5, amp, -amp)
        elif self.waveform == "Triangle":
            buf = amp * (4.0 * np.where(t < 0.5, t, 1.0 - t) - 1.0)
        else:
            buf = amp * (2.0 * t - 1.0)

        outdata[:, 0] = buf.astype(np.float32)
        self._phase = (self._phase + frames * dt) % 1.0

    def start(self):
        if self._stream is not None:
            return
        self._stream = sd.OutputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=self.BLOCKSIZE,
            latency="low",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class IRTouchThread(QThread):
    touch_pos = pyqtSignal(int, int)
    touch_down = pyqtSignal()
    touch_up = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, device_path, parent=None):
        super().__init__(parent)
        self.device_path = device_path
        self._running = True

    def run(self):
        try:
            device = InputDevice(self.device_path)
        except Exception as exc:
            self.error.emit(str(exc))
            return

        x, y, touching = 0, 0, False
        for event in device.read_loop():
            if not self._running:
                break
            if event.type == ecodes.EV_ABS:
                if event.code == ecodes.ABS_MT_POSITION_X:
                    x = event.value
                elif event.code == ecodes.ABS_MT_POSITION_Y:
                    y = event.value
            elif event.type == ecodes.EV_KEY and event.code == ecodes.BTN_TOUCH:
                if event.value == 1:
                    touching = True
                    self.touch_down.emit()
                else:
                    touching = False
                    self.touch_up.emit()
            elif event.type == ecodes.EV_SYN and touching:
                self.touch_pos.emit(x, y)

    def stop(self):
        self._running = False
        self.quit()
        self.wait(1000)


class ExperimentDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Participant Setup")
        self.setModal(True)
        layout = QFormLayout(self)

        self.participant = QLineEdit()
        self.participant.setPlaceholderText("e.g. 01")
        self.condition = QComboBox()
        self.condition.addItems(["T", "V"])

        layout.addRow("Participant number:", self.participant)
        layout.addRow("Condition:", self.condition)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _accept(self):
        if not self.participant.text().strip():
            QMessageBox.warning(self, "Missing input", "Please enter a participant number.")
            return
        self.accept()

    def values(self):
        return self.participant.text().strip(), self.condition.currentText().strip()


class ExperimentCanvas(QWidget):
    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self.owner = owner
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(900, 500)
        self.setStyleSheet("background-color: #111118; border: 1px solid #333;")
        self.phase = "cal"
        self.cal_step = 0
        self.cursor_x = -1
        self.cursor_y = -1
        self._display_w = 0
        self._display_h = 0
        self._offset_x = 0
        self._offset_y = 0
        self.case1_black_quadrant = random.randrange(4)
        self.case2_amplitude_px = 270.0
        self.case2_cycles = 1.0
        self.case2_thickness_px = 18.0
        self.case2_recording = False
        self.case2_distances = []
        self.case3_active = False
        self.case3_sound_time = None
        self.case3_trigger_pending = False
        self.case3_trial_armed_time = None
        self.case3_trigger_delay = None
        self.case3_dot_x = 0.0
        self.case3_dot_y = 0.0
        self.case3_motion_t = 0.0
        self.case3_motion_dir = 1.0

        self._case2_pixmaps = {}
        self._case2_active_pixmap = None
        self._load_case2_images()

    def _load_case2_images(self):
        folder = get_images_folder()
        print(f"[DEBUG] Looking for option images in: {folder}")
        for idx in range(3):
            path = os.path.join(folder, f"option{idx+1}.png")
            if os.path.exists(path):
                pm = QPixmap(path)
                if not pm.isNull():
                    self._case2_pixmaps[idx] = pm
                    print(f"[OK] Loaded {path}")
                else:
                    print(f"[ERROR] Corrupt image: {path}")
            else:
                print(f"[ERROR] Missing: {path}")

    def set_case2_option(self, index):
        if index in self._case2_pixmaps:
            self._case2_active_pixmap = self._case2_pixmaps[index]
        else:
            self._case2_active_pixmap = None
        self.update()

    def sizeHint(self):
        return QSize(100, 100)

    def minimumSizeHint(self):
        return QSize(100, 100)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_geometry()
        self.update()

    def _update_geometry(self):
        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            return
        scale = min(w / LOGICAL_WIDTH, h / LOGICAL_HEIGHT)
        self._display_w = int(LOGICAL_WIDTH * scale)
        self._display_h = int(LOGICAL_HEIGHT * scale)
        self._offset_x = (w - self._display_w) // 2
        self._offset_y = (h - self._display_h) // 2

    def image_bounds(self):
        return self._offset_x, self._offset_y, self._display_w, self._display_h

    def image_to_widget(self, img_x, img_y):
        if self._display_w <= 0 or self._display_h <= 0:
            return -1, -1
        wx = self._offset_x + int(img_x * self._display_w / LOGICAL_WIDTH)
        wy = self._offset_y + int(img_y * self._display_h / LOGICAL_HEIGHT)
        return wx, wy

    def widget_to_image(self, widget_x, widget_y):
        if self._display_w <= 0 or self._display_h <= 0:
            return -1, -1
        ix = int((widget_x - self._offset_x) * LOGICAL_WIDTH / self._display_w)
        iy = int((widget_y - self._offset_y) * LOGICAL_HEIGHT / self._display_h)
        return ix, iy

    def touch_to_image(self, tactile_x, tactile_y):
        if self._display_w <= 0 or self._display_h <= 0:
            return -1, -1
        if getattr(self.owner, "homography", None) is not None:
            img_x, img_y = apply_homography(self.owner.homography, tactile_x, tactile_y)
            return int(img_x), int(img_y)
        return int(tactile_x), int(tactile_y)

    def case2_curve_y(self, x):
        x = clamp_value(x, 0, LOGICAL_WIDTH)
        amp = float(self.case2_amplitude_px)
        cycles = float(self.case2_cycles)
        return (LOGICAL_HEIGHT / 2.0) + amp * math.sin(2.0 * math.pi * cycles * (x / LOGICAL_WIDTH))

    def case2_is_black(self, x, y):
        return abs(y - self.case2_curve_y(x)) <= (self.case2_thickness_px / 2.0)

    def case2_distance(self, x, y):
        return abs(y - self.case2_curve_y(x))

    def case3_tick(self, dt):
        if self.phase != "track" or self.owner.current_case != 3:
            return
        self.case3_motion_t += dt
        speed = self.owner.case3_speed_px_s
        margin = self.owner.case3_margin_px
        left_edge = margin
        right_edge = LOGICAL_WIDTH - margin
        next_x = self.case3_dot_x + (speed * dt * self.case3_motion_dir)
        if next_x >= right_edge:
            next_x = right_edge
            self.case3_motion_dir = -1.0
        elif next_x <= left_edge:
            next_x = left_edge
            self.case3_motion_dir = 1.0
        self.case3_dot_x = next_x
        self.case3_dot_y = LOGICAL_HEIGHT / 2.0
        if (
            self.case3_trigger_pending
            and self.case3_trial_armed_time is not None
            and time.perf_counter() >= self.case3_trial_armed_time
            and 300.0 <= self.case3_dot_x <= 1600.0
        ):
            self.case3_trigger_pending = False
            self.owner.trigger_case3_sound()

    def start_case2_recording(self):
        self.case2_recording = True
        self.case2_distances = []

    def stop_case2_recording(self):
        self.case2_recording = False
        return self.case2_distances[:]

    def arm_case3_trial(self):
        self.case3_trigger_pending = True
        self.case3_trial_armed_time = time.perf_counter() + random.uniform(3.0, 5.0)
        self.case3_sound_time = None

    def cancel_case3_trial(self):
        self.case3_trigger_pending = False
        self.case3_trial_armed_time = None
        self.case3_sound_time = None

    def mark_case3_response(self):
        if self.case3_sound_time is None:
            return None
        rt_ms = (time.perf_counter() - self.case3_sound_time) * 1000.0
        self.case3_sound_time = None
        return rt_ms

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#111118"))

        if self._display_w <= 0 or self._display_h <= 0:
            painter.end()
            return

        painter.translate(self._offset_x, self._offset_y)
        painter.scale(self._display_w / LOGICAL_WIDTH, self._display_h / LOGICAL_HEIGHT)

        painter.fillRect(QRectF(0, 0, LOGICAL_WIDTH, LOGICAL_HEIGHT), QColor("#ffffff"))
        painter.setPen(QPen(QColor(40, 40, 40), 4))
        painter.drawRect(QRectF(0, 0, LOGICAL_WIDTH, LOGICAL_HEIGHT))

        if self.phase == "cal":
            self._paint_calibration(painter)
        elif self.owner.current_case == 1:
            self._paint_case1(painter)
        elif self.owner.current_case == 2:
            self._paint_case2(painter)
        elif self.owner.current_case == 3:
            self._paint_case3(painter)

        if self.phase == "track" and self.cursor_x >= 0 and self.cursor_y >= 0:
            self._paint_cursor(painter)

        painter.end()

    def _paint_calibration(self, painter):
        labels = ["1\nTL", "2\nTR", "3\nBL", "4\nBR"]
        corners = [
            (0, 0),
            (LOGICAL_WIDTH - 1, 0),
            (0, LOGICAL_HEIGHT - 1),
            (LOGICAL_WIDTH - 1, LOGICAL_HEIGHT - 1),
        ]
        for idx, (x, y) in enumerate(corners):
            if idx < self.cal_step:
                color, radius = QColor(30, 210, 100), 18
            elif idx == self.cal_step:
                color, radius = QColor(255, 40, 40), 24
            else:
                color, radius = QColor(160, 40, 40), 16
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawEllipse(QPointF(x, y), radius, radius)
            painter.setFont(QFont("Monospace", 10, QFont.Bold))
            painter.drawText(x - radius, y - radius, 2 * radius, 2 * radius, Qt.AlignCenter, labels[idx])

    def _paint_case1(self, painter):
        half_w = LOGICAL_WIDTH / 2.0
        half_h = LOGICAL_HEIGHT / 2.0
        quads = [
            QRectF(0, 0, half_w, half_h),
            QRectF(half_w, 0, half_w, half_h),
            QRectF(0, half_h, half_w, half_h),
            QRectF(half_w, half_h, half_w, half_h),
        ]
        painter.setPen(QPen(QColor(30, 30, 30), 4))
        for idx, rect in enumerate(quads):
            painter.fillRect(rect, QColor("#000000") if idx == self.case1_black_quadrant else QColor("#ffffff"))
            painter.drawRect(rect)
        painter.setPen(QPen(QColor(75, 75, 75), 8))
        painter.drawLine(int(half_w), 0, int(half_w), LOGICAL_HEIGHT)
        painter.drawLine(0, int(half_h), LOGICAL_WIDTH, int(half_h))

    def _paint_case2(self, painter):
        if self._case2_active_pixmap is not None:
            painter.drawPixmap(0, 0, self._case2_active_pixmap)
        else:
            painter.fillRect(0, 0, LOGICAL_WIDTH, LOGICAL_HEIGHT, QColor("#eeeeee"))

    def _paint_case3(self, painter):
        painter.setPen(QPen(QColor(160, 160, 160), 3, Qt.DashLine))
        mid_y = int(LOGICAL_HEIGHT / 2)
        painter.drawLine(0, mid_y, LOGICAL_WIDTH, mid_y)
        radius = self.owner.case3_dot_radius_px
        painter.setBrush(QBrush(QColor(45, 120, 255)))
        painter.setPen(QPen(QColor(255, 255, 255), 4))
        painter.drawEllipse(QPointF(self.case3_dot_x, self.case3_dot_y), radius, radius)

    def _paint_cursor(self, painter):
        painter.setBrush(QBrush(QColor(255, 40, 40)))
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawEllipse(QPointF(self.cursor_x, self.cursor_y), 9, 9)


class TactileFeedbackNode(Node):
    def __init__(self):
        super().__init__("char_exp_node")
        self.robot_pub = self.create_publisher(Vector3, "/unity_robot/position_command", 10)
        self.pixel_pub = self.create_publisher(Vector3, "/unity_robot/pixel_position", 10)
        self.sound_pub = self.create_publisher(Vector3, "/unity_robot/sound_details", 10)
        self.get_logger().info("Char experiment ROS2 node initialized")

    def publish_position(self, x, y, z):
        msg = Vector3()
        msg.x = float(x)
        msg.y = float(y)
        msg.z = float(z)
        self.robot_pub.publish(msg)

    def publish_pixel_position(self, px_x, px_y):
        msg = Vector3()
        msg.x = float(px_x)
        msg.y = float(px_y)
        msg.z = 0.0
        self.pixel_pub.publish(msg)

    def publish_sound_details(self, amplitude, frequency, waveform):
        msg = Vector3()
        msg.x = float(np.clip(amplitude, 0.0, 1.0))
        msg.y = float(frequency)
        msg.z = float(waveform_index(waveform))
        self.sound_pub.publish(msg)


class ROS2Thread(QThread):
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node

    def run(self):
        try:
            rclpy.spin(self.ros_node)
        except Exception as exc:
            print(f"[ROS2] Error: {exc}")


class MainWindow(QMainWindow):
    # Mapping constants (same as tactile_feedback_ros2.py)
    TACTILE_WIDTH = 1920
    TACTILE_HEIGHT_MAX = 900

    def __init__(self, device_path, ros_node, participant_number, condition_number):
        super().__init__()
        self.ros_node = ros_node
        self.participant_number = participant_number
        self.condition_number = condition_number
        self.data_dir = Path(__file__).resolve().parent
        self.path_following_path = self.data_dir / "Path_following.csv"
        self.response_time_path = self.data_dir / "Response_time.csv"
        self.current_case = 1
        self.case3_speed_px_s = 300.0
        self.case3_margin_px = 180.0
        self.case3_dot_radius_px = 52.0
        self.case3_repeat_counter = 1
        self.case2_repeat_counter = 1
        self._touching = False
        self._last_touch_point = (-1, -1)
        self.homography = None
        self._cal_ir_points = []
        self._collecting = False
        self._case2_recording = False
        self._sound = SoundEngine()
        self._sound.start()
        self._sound_active = False
        self._last_sound_freq = CASE1_FREQ
        self._last_sound_waveform = CASE1_WAVEFORM
        self._case3_pending = False
        self._case3_record_trial = False
        self._case3_trigger_time = None
        self._sound_on_wall_time = None

        self._last_distance_update = 0.0
        self._last_pos_update = 0.0

        self._case3_timer = QTimer(self)
        self._case3_timer.timeout.connect(self._tick_case3)
        self._last_case3_tick = 0.0

        print(f"[STARTUP] Script location : {Path(__file__).resolve()}")
        print(f"[STARTUP] Data directory  : {self.data_dir}")
        print(f"[STARTUP] Path_following.csv : {self.path_following_path}")
        print(f"[STARTUP] Response_time.csv  : {self.response_time_path}")

        self.setWindowTitle("Char Experiment - ROS2")
        self.resize(1280, 760)
        self.setMinimumSize(900, 640)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        controls = QFrame()
        controls.setStyleSheet("background-color: #10131c; border: 1px solid #2c3142; border-radius: 6px;")
        grid = QHBoxLayout(controls)
        grid.setContentsMargins(12, 8, 12, 8)
        grid.setSpacing(10)

        self.combo_case = QComboBox()
        self.combo_case.addItems(["Case 1", "Case 2", "Case 3"])
        self.combo_case.currentIndexChanged.connect(self._on_case_changed)

        self.btn_shuffle = QPushButton("Shf")
        self.btn_shuffle.clicked.connect(self._shuffle_case1)

        self.btn_case2_record = QPushButton("Rec: OFF")
        self.btn_case2_record.setCheckable(True)
        self.btn_case2_record.clicked.connect(self._toggle_case2_recording)

        self.btn_random = QPushButton("Rnd")
        self.btn_random.setCheckable(True)
        self.btn_random.clicked.connect(lambda: self._arm_case3_trial(False))
        self.btn_case3_record = QPushButton("Record")
        self.btn_case3_record.setCheckable(True)
        self.btn_case3_record.clicked.connect(lambda: self._arm_case3_trial(True))

        self.lbl_case = self._make_label("C1", "#ffaa00")
        self.lbl_wave = self._make_label("W", "#d7dcff")

        self.combo_wave = QComboBox()
        self.combo_wave.addItems([
            "1: A:150 f:1.00",
            "2: A:250 f:2.00",
            "3: A:350 f:1.50",
        ])
        self.combo_wave.currentIndexChanged.connect(self._sync_case2_params)

        grid.addWidget(self.lbl_case)
        grid.addWidget(self.combo_case)
        grid.addWidget(self.btn_shuffle)
        grid.addWidget(self.btn_case2_record)
        grid.addWidget(self.btn_random)
        grid.addWidget(self.btn_case3_record)
        grid.addWidget(self.lbl_wave)
        grid.addWidget(self.combo_wave)

        self.lbl_participant = self._make_label(f"P:{participant_number}", "#d7dcff")
        self.lbl_condition = self._make_label(f"C:{condition_number}", "#d7dcff")
        self.lbl_sound = self._make_label("Off", "#777777")
        self.lbl_pos = self._make_label("Px: --", "#999999")
        self.lbl_case2_distance = self._make_label("D: --", "#dcdcff")
        self.lbl_case3_rt = self._make_label("RT: --", "#dcdcff")
        for label in (
            self.lbl_participant,
            self.lbl_condition,
            self.lbl_sound,
            self.lbl_pos,
            self.lbl_case2_distance,
            self.lbl_case3_rt,
        ):
            grid.addWidget(label)

        self.lbl_case2_distance.setVisible(False)
        self.lbl_case3_rt.setVisible(False)

        root.addWidget(controls)

        self.canvas = ExperimentCanvas(self)
        root.addWidget(self.canvas, stretch=1)

        self.setStatusBar(QStatusBar())
        self._dark_theme()

        self._ensure_data_files()
        self._configure_case_controls()

        self._ir = IRTouchThread(device_path)
        self._ir.touch_pos.connect(self._on_pos)
        self._ir.touch_down.connect(self._on_down)
        self._ir.touch_up.connect(self._on_up)
        self._ir.error.connect(self._on_ir_error)
        self._ir.start()

        self.canvas.phase = "cal"
        self.canvas.update()
        self.statusBar().showMessage("Calibrate dots 1-4.")

    def _map_position_to_ros2(self, tactile_x, tactile_y):
        """Map pixel coordinates to robot space (same as tactile_feedback_ros2.py)."""
        cx = min(max(tactile_x, 0), self.TACTILE_WIDTH)
        cy = min(max(tactile_y, 0), self.TACTILE_HEIGHT_MAX)
        return (0.8 - (1.6 * (cx / self.TACTILE_WIDTH)),
                0.0,
                -1.05 + (0.75 * (cy / self.TACTILE_HEIGHT_MAX)))

    def _ensure_data_files(self):
        if not self.path_following_path.exists():
            with self.path_following_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "timestamp",
                    "participant_number",
                    "condition_number",
                    "repeat_number",
                    "trial_type",
                    "wave_preset",
                    "average_distance_px",
                    "max_distance_px",
                    "notes",
                ])
            print(f"[STARTUP] Created {self.path_following_path}")
        if not self.response_time_path.exists():
            with self.response_time_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "timestamp",
                    "participant_number",
                    "condition_number",
                    "repeat_number",
                    "case",
                    "trial_type",
                    "reaction_time_ms",
                    "sound_on_timestamp",
                    "release_timestamp",
                    "notes",
                ])
            print(f"[STARTUP] Created {self.response_time_path}")

    def _append_row(self, path, row):
        print(f"[LOG] Writing to {path}: {row}")
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(row)
            handle.flush()

    def _make_label(self, text, color):
        label = QLabel(text)
        label.setFont(QFont("Monospace", 11))
        label.setStyleSheet(f"color: {color}; background: transparent;")
        return label

    def _dark_theme(self):
        self.setStyleSheet(
            """
            QMainWindow { background-color: #0c0c16; }
            QWidget { background-color: #0c0c16; color: #e0e0e0; }
            QPushButton { background-color: #1d2540; color: #eef; border: 1px solid #405070; padding: 6px 12px; border-radius: 4px; }
            QPushButton:checked { background-color: #3355aa; }
            QStatusBar { background-color: #16213e; color: #777; font-size: 10px; padding: 2px 8px; }
            QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit { background-color: #171d30; color: #e0e0ff; border: 1px solid #334; border-radius: 3px; padding: 2px 4px; }
            """
        )

    def _configure_case_controls(self):
        self.btn_shuffle.setVisible(True)
        self.btn_case2_record.setVisible(False)
        self.btn_random.setVisible(False)
        self.btn_case3_record.setVisible(False)
        self.lbl_wave.setVisible(False)
        self.combo_wave.setVisible(False)
        self._sync_case2_params()

    def _on_case_changed(self, index):
        if self.current_case == 2 and self._case2_recording:
            self._toggle_case2_recording()

        self._case3_timer.stop()
        self.current_case = index + 1
        self.canvas.phase = "track"
        self.canvas.cursor_x = -1
        self.canvas.cursor_y = -1
        self.canvas.cancel_case3_trial()
        self._set_sound(False)

        self.btn_shuffle.setVisible(self.current_case == 1)
        self.btn_case2_record.setVisible(self.current_case == 2)
        self.btn_random.setVisible(self.current_case == 3)
        self.btn_case3_record.setVisible(self.current_case == 3)
        self.lbl_wave.setVisible(self.current_case == 2)
        self.combo_wave.setVisible(self.current_case == 2)
        self.lbl_case2_distance.setVisible(self.current_case == 2)
        self.lbl_case3_rt.setVisible(self.current_case == 3)

        if self.current_case == 1:
            self.lbl_case.setText("C1")
        elif self.current_case == 2:
            self.lbl_case.setText("C2")
        else:
            self.lbl_case.setText("C3")

        if self.current_case == 3:
            self._last_case3_tick = time.perf_counter()
            self._case3_timer.start(16)

        self.canvas.update()

    def _sync_case2_params(self):
        preset = self.combo_wave.currentIndex()
        if preset == 0:
            amp, freq = 150.0, 1.0
        elif preset == 1:
            amp, freq = 250.0, 2.0
        else:
            amp, freq = 350.0, 1.5
        self.canvas.case2_amplitude_px = amp
        self.canvas.case2_cycles = freq
        self.canvas.case2_thickness_px = 120.0
        self.canvas.set_case2_option(preset)
        self.canvas.update()

    def _shuffle_case1(self):
        if self.current_case == 1:
            self.canvas.case1_black_quadrant = random.randrange(4)
            self.canvas.update()
            self.statusBar().showMessage("Case 1 shuffled.")

    def _toggle_case2_recording(self):
        if self.current_case != 2:
            return
        if not self._case2_recording:
            self._case2_recording = True
            self.canvas.start_case2_recording()
            self.btn_case2_record.setText("Rec: ON")
            self.btn_case2_record.setChecked(True)
            self.statusBar().showMessage("C2 recording ...")
            print("[REC] Case 2 recording started")
        else:
            self._case2_recording = False
            distances = self.canvas.stop_case2_recording()
            avg_distance = float(np.mean(distances)) if distances else 0.0
            max_distance = float(np.max(distances)) if distances else 0.0
            row = [
                datetime.now().isoformat(timespec="milliseconds"),
                self.participant_number,
                self.condition_number,
                self.case2_repeat_counter,
                "path_follow",
                str(self.combo_wave.currentIndex() + 1),
                f"{avg_distance:.3f}",
                f"{max_distance:.3f}",
                "manual stop",
            ]
            self._append_row(self.path_following_path, row)
            self.case2_repeat_counter += 1
            self.btn_case2_record.setText("Rec: OFF")
            self.btn_case2_record.setChecked(False)
            self.statusBar().showMessage("C2 trial saved")
            print("[REC] Case 2 trial saved to Path_following.csv")

    def _arm_case3_trial(self, record_trial):
        if self.current_case != 3:
            return
        self.btn_random.setChecked(not record_trial)
        self.btn_case3_record.setChecked(record_trial)
        self.canvas.arm_case3_trial()
        self._case3_pending = True
        self._case3_record_trial = bool(record_trial)
        self.lbl_case3_rt.setText("RT: waiting")
        self.statusBar().showMessage("C3 armed")

    def trigger_case3_sound(self):
        self._case3_pending = False
        self.canvas.case3_sound_time = time.perf_counter()
        self._sound_on_wall_time = time.time()
        self._sound.freq = CASE1_FREQ
        self._sound.amplitude = CASE1_AMPLITUDE
        self._sound.waveform = CASE1_WAVEFORM
        self._set_sound(True)
        self._last_sound_freq = CASE1_FREQ
        self._last_sound_waveform = CASE1_WAVEFORM
        self.ros_node.publish_sound_details(1.0, CASE1_FREQ, CASE1_WAVEFORM)
        self.statusBar().showMessage("C3 sound")

    def _set_sound(self, on):
        if self._sound.active == on:
            return
        self._sound.active = on
        self._sound_active = on
        if on:
            self.lbl_sound.setText("On")
            self.lbl_sound.setStyleSheet("color: #00ff88; font-weight: bold; background: transparent;")
        else:
            self.lbl_sound.setText("Off")
            self.lbl_sound.setStyleSheet("color: #777777; background: transparent;")

    def _on_ir_error(self, msg):
        self.statusBar().showMessage(f"IR device error: {msg}")

    def _on_down(self):
        self._touching = True
        if self.canvas.phase == "cal":
            self._collecting = True
            self._cal_samples = []

    def _on_up(self):
        self._touching = False
        if self.canvas.phase == "cal":
            if self._collecting and self._cal_samples:
                arr = np.array(self._cal_samples, dtype=float)
                self._collecting = False
                self._accept_cal_point(int(np.median(arr[:, 0])), int(np.median(arr[:, 1])))
            return

        if self.current_case == 3 and self.canvas.case3_sound_time is not None:
            rt_ms = self.canvas.mark_case3_response()
            if rt_ms is not None:
                row = [
                    datetime.now().isoformat(timespec="milliseconds"),
                    self.participant_number,
                    self.condition_number,
                    self.case3_repeat_counter,
                    3,
                    "reaction_time",
                    f"{rt_ms:.3f}",
                    datetime.fromtimestamp(self._sound_on_wall_time).isoformat(timespec="milliseconds") if self._sound_on_wall_time else "",
                    datetime.now().isoformat(timespec="milliseconds"),
                    "finger lift-off",
                ]
                if self._case3_record_trial:
                    self._append_row(self.response_time_path, row)
                    self.case3_repeat_counter += 1
                    print("[CASE3] RT trial saved to Response_time.csv")
                self.lbl_case3_rt.setText(f"RT: {rt_ms:.1f} ms")
                if self._case3_record_trial:
                    self.statusBar().showMessage(f"Case 3 RT saved: {rt_ms:.1f} ms")
                else:
                    self.statusBar().showMessage(f"Case 3 RT preview: {rt_ms:.1f} ms")
                self.btn_random.setChecked(False)
                self.btn_case3_record.setChecked(False)
        self._set_sound(False)
        self.ros_node.publish_sound_details(0.0, self._last_sound_freq, self._last_sound_waveform)
        self.canvas.cursor_x = -1
        self.canvas.cursor_y = -1
        self.canvas.update()

    def _on_pos(self, x, y):
        if self.canvas.phase == "cal":
            if self._collecting:
                self._cal_samples.append((x, y))
            return

        self._last_touch_point = (x, y)

        img_x, img_y = self.canvas.touch_to_image(x, y)
        self.canvas.cursor_x = img_x
        self.canvas.cursor_y = img_y

        if self.current_case == 1:
            self._track_case1(img_x, img_y)
        elif self.current_case == 2:
            self._track_case2(img_x, img_y)
        elif self.current_case == 3:
            self._track_case3(img_x, img_y)

        self.canvas.update()

    def _accept_cal_point(self, ir_x, ir_y):
        self._cal_ir_points.append((ir_x, ir_y))
        self.canvas.cal_step += 1
        self.canvas.update()
        if self.canvas.cal_step >= 4:
            try:
                self.homography = compute_homography(
                    self._cal_ir_points,
                    [(0, 0), (LOGICAL_WIDTH - 1, 0), (0, LOGICAL_HEIGHT - 1), (LOGICAL_WIDTH - 1, LOGICAL_HEIGHT - 1)],
                )
            except Exception as exc:
                QMessageBox.critical(self, "Calibration error", f"Calibration failed: {exc}\nRestart the app and calibrate again.")
                QApplication.instance().quit()
                return
            self.canvas.phase = "track"
            self.statusBar().showMessage("Tracking active. Touch the frame to start tracking.")

    def _track_case1(self, img_x, img_y):
        if img_x < 0 or img_y < 0:
            self._set_sound(False)
            return
        quadrant = (1 if img_x >= (LOGICAL_WIDTH / 2.0) else 0) + (2 if img_y >= (LOGICAL_HEIGHT / 2.0) else 0)
        black = quadrant == self.canvas.case1_black_quadrant
        self._set_sound(black)
        if black:
            self._sound.freq = CASE1_FREQ
            self._sound.amplitude = CASE1_AMPLITUDE
            self._sound.waveform = CASE1_WAVEFORM
            self._last_sound_freq = CASE1_FREQ
            self._last_sound_waveform = CASE1_WAVEFORM

        x_ros, y_ros, z_ros = self._map_position_to_ros2(img_x, img_y)
        self.ros_node.publish_position(x_ros, y_ros, z_ros)
        self.ros_node.publish_pixel_position(img_x, img_y)
        self.ros_node.publish_sound_details(1.0 if black else 0.0, CASE1_FREQ, CASE1_WAVEFORM)
        self.lbl_pos.setText(f"Pos: ({img_x:4d}, {img_y:4d})")

    def _track_case2(self, img_x, img_y):
        if img_x < 0 or img_y < 0:
            self._set_sound(False)
            return

        distance = self.canvas.case2_distance(img_x, img_y)
        half = self.canvas.case2_thickness_px / 2.0
        hyster = 8.0

        if self._sound_active:
            black = distance <= (half + hyster)
        else:
            black = distance <= half

        if black != self._sound_active:
            self._set_sound(black)
            if black:
                self._sound.freq = CASE1_FREQ
                self._sound.amplitude = CASE1_AMPLITUDE
                self._sound.waveform = CASE1_WAVEFORM
                self._last_sound_freq = CASE1_FREQ
                self._last_sound_waveform = CASE1_WAVEFORM
            self.ros_node.publish_sound_details(1.0 if black else 0.0, CASE1_FREQ, CASE1_WAVEFORM)

        x_ros, y_ros, z_ros = self._map_position_to_ros2(img_x, img_y)
        self.ros_node.publish_position(x_ros, y_ros, z_ros)
        self.ros_node.publish_pixel_position(img_x, img_y)

        now = time.perf_counter()
        if now - self._last_distance_update >= 0.5:
            self._last_distance_update = now
            self.lbl_case2_distance.setText(f"D:{distance:.1f}")
        if now - self._last_pos_update >= 0.2:
            self._last_pos_update = now
            self.lbl_pos.setText(f"Pos: ({img_x:4d}, {img_y:4d})")

        if self._case2_recording:
            self.canvas.case2_distances.append(distance)

    def _track_case3(self, img_x, img_y):
        x_ros, y_ros, z_ros = self._map_position_to_ros2(img_x, img_y)
        self.ros_node.publish_position(x_ros, y_ros, z_ros)
        self.ros_node.publish_pixel_position(int(self.canvas.case3_dot_x), int(self.canvas.case3_dot_y))
        self.ros_node.publish_sound_details(1.0 if self._sound.active else 0.0, self._last_sound_freq, self._last_sound_waveform)
        self.lbl_pos.setText(f"Pos: ({img_x:4d}, {img_y:4d})")

    def _tick_case3(self):
        now = time.perf_counter()
        dt = now - self._last_case3_tick
        self._last_case3_tick = now
        self.canvas.case3_tick(dt)
        self.canvas.update()

    def closeEvent(self, event):
        if self._case2_recording:
            self._toggle_case2_recording()
        try:
            self._ir.stop()
        except Exception:
            pass
        try:
            self._sound.stop()
        except Exception:
            pass
        self._case3_timer.stop()
        super().closeEvent(event)


def parse_args():
    parser = argparse.ArgumentParser(description="Char experiment ROS2 tactile app")
    parser.add_argument("--device", default="", help="evdev path e.g. /dev/input/event20")
    return parser.parse_args()


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

    app = QApplication(sys.argv)
    dialog = ExperimentDialog()
    if dialog.exec_() != QDialog.Accepted:
        rclpy.shutdown()
        return 0

    participant_number, condition_number = dialog.values()
    ros_node = TactileFeedbackNode()
    window = MainWindow(device_path, ros_node, participant_number, condition_number)
    window.show()

    ros2_thread = ROS2Thread(ros_node)
    ros2_thread.start()

    try:
        exit_code = app.exec_()
    finally:
        try:
            ros_node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        print("Goodbye")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())