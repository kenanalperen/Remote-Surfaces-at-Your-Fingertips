# Remote Surfaces at Your Fingertips

**Electrovibration-Based Tactile Feedback for Robot Teleoperation via Touchscreen Interfaces**

![Tactile Feedback Demo](https://github.com/kenanalperen/Remote-Surfaces-at-Your-Fingertips/raw/main/Tactile_Feedback.gif)

[▶ Watch Full Demo Video](https://uweacuk-my.sharepoint.com/:v:/g/personal/alperen_kenan_uwe_ac_uk/IQCmrqgfMpsmQo4FHIiwfGfsAWtH6rBSz5ab69s8c_FVUs8?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D&e=J1QKDK)

> **Associated Paper:** *Remote Surfaces at Your Fingertips: Electrovibration-Based Tactile Feedback for Robot Teleoperation via Touchscreen Interfaces*, accepted at **IEEE Telepresence 2026**, Bristol, UK.

---

## Overview

Real-time haptic awareness is critical for teleoperated tasks involving physical contact, but existing feedback methods have real drawbacks: kinesthetic feedback becomes unstable during stiff surface contact and is sensitive to communication delay, while visual force cues add cognitive load and reduce situational awareness.

This project implements a **touchscreen-based teleoperation interface** that reflects remote robot–environment interaction forces to the operator's fingertip in real time using **electrovibration** — a technique that modulates the electrostatic friction force on a conductive touch surface via an alternating voltage, with no mechanically moving parts and minimal signal latency.

A **user study (N = 21)** compared this tactile feedback (TF) condition against a conventional visual bar-graph feedback (VF) condition across both controlled characterisation tasks and realistic nuclear-inspection-style evaluation tasks (surface defect detection, multi-surface swabbing, and obstacle avoidance) using a simulated UR10 arm in Unity.

---

## Key Results

| Metric | Visual Feedback (VF) | Tactile Feedback (TF) | Result |
|---|---|---|---|
| Response time | 466.92 ms | 395.24 ms | **↓ 15.35%**, *p* = .002, *d* = 0.96 |
| Sense of telepresence | 56.36 / 100 | 73.87 / 100 | **↑ 31%**, *p* < .001, *d* = 0.90 |
| Path-following accuracy | 65.96 px avg. error | 63.84 px avg. error | No significant difference |
| Cognitive load (NASA-TLX) | 24.78 | 27.04 | No significant difference |
| Usability (SUS) | 79.88 | 78.39 | No significant difference |

Tactile feedback significantly **reduced response latency** and **increased sense of telepresence**, particularly the feeling of direct control over the robot, while imposing comparable workload and usability to visual feedback. Benefits were most pronounced in contact-rich, multi-surface manipulation tasks such as swabbing and defect detection.

---

## How It Works

The interface maps the operator's finger position on a touchscreen to the robot end-effector's coordinates, using a live camera feed for context. As the robot contacts the remote surface, interaction forces are classified in real time and translated into a corresponding electrovibration voltage (amplitude, frequency, and waveform), delivered to the operator's fingertip.

Because a high-voltage signal can't be applied directly to a commercial touchscreen, the physical interface is built from **three stacked layers**:

1. A standard monitor providing visual output
2. A capacitive conductive touchscreen delivering the electrovibration effect
3. An IR touch frame for finger position tracking

These are integrated into a single unit using laser-cut acrylic components. Voltage signals are generated and amplified via a Texas Instruments DRV2667EVM-CT Touch-Path Evaluation Module, with the primary signal being a 200 Vpp, 120 Hz square wave (varied per surface in the multi-surface condition).

---

## Figures

| | | |
|---|---|---|
| ![Image Processing 1](https://github.com/kenanalperen/Remote-Surfaces-at-Your-Fingertips/raw/main/image_process_1.png) | ![Image Processing 2](https://github.com/kenanalperen/Remote-Surfaces-at-Your-Fingertips/raw/main/image_process_2_v2.png) | ![Image Processing 3](https://github.com/kenanalperen/Remote-Surfaces-at-Your-Fingertips/raw/main/image_process_3_v6.png) |

*Image processing pipeline used to detect surface features and drive tactile feedback during the evaluation tasks.*

| | | |
|---|---|---|
| ![Interface Option 1](https://github.com/kenanalperen/Remote-Surfaces-at-Your-Fingertips/raw/main/option1.png) | ![Interface Option 2](https://github.com/kenanalperen/Remote-Surfaces-at-Your-Fingertips/raw/main/option2.png) | ![Interface Option 3](https://github.com/kenanalperen/Remote-Surfaces-at-Your-Fingertips/raw/main/option3.png) |

*Interface hardware layer configurations explored during development.*

---

## Repository Contents

| File | Description |
|---|---|
| `Tactile_Feedback.gif` / `.mp4` | Demo of the electrovibration touchscreen interface in use |
| `char_exp.py` | Script for the characterisation experiments (path-following accuracy and response time measurement) |
| `tactile_feedback_ros2.py` | ROS2 script for generating and mapping tactile feedback signals to robot–environment interaction data |
| `image_process_*.png` | Image processing pipeline used for the multi-surface swabbing and defect detection tasks |
| `option*.png` | Candidate hardware/interface layer configurations explored during design |

---

## Hypotheses Tested

- **H1** — Electrovibration-based tactile feedback can convey force outputs in real time and is perceivable by participants. ✅ Supported
- **H2** — TF yields faster response times than VF. ✅ Supported
- **H3** — TF yields a significantly higher sense of telepresence than VF. ✅ Supported
- **H4** — TF yields significantly lower cognitive workload than VF. ⚠️ Not supported (no significant difference)

---

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{kenan2026remote,
  title     = {Remote Surfaces at Your Fingertips: Electrovibration-Based Tactile Feedback for Robot Teleoperation via Touchscreen Interfaces},
  author    = {Kenan, Alperen and Garc{\'\i}a C{\'a}rdenas, Juan Jos{\'e} and
               Tapus, Adriana and Bremner, Paul and Giuliani, Manuel},
  booktitle = {IEEE Telepresence 2026},
  address   = {Bristol, UK},
  year      = {2026}
}
```

---

## Acknowledgements

This work was supported by the European Commission's HORIZON.1.2 Marie Skłodowska-Curie Actions (MSCA) under Grant Agreement No. **101072634**, project [RAICAM](https://raicam.eu/), and by **UK Research and Innovation (UKRI)** grant number **EP/X025977/1**.
