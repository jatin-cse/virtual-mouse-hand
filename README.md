# Hand Gesture Virtual Mouse (Python)

Control your computer mouse using **hand gestures** in real time through a webcam.  
This project uses **Python, OpenCV, MediaPipe, and PyAutoGUI** to turn your hand into a virtual mouse.

---

## Features
- Smooth mouse movement using index finger
- Left click using thumb + index finger pinch
- Right click using thumb + index + middle finger pinch
- Scroll up and down using index & middle fingers
- Full screen coverage (no dead zones)
- Stable and beginner-friendly logic

---

## Gesture Controls

| Gesture | Action |
|------|------|
| ☝ Index finger move | Move mouse |
| 🤏 Thumb + Index | Left click |
| 🤏 Thumb + Index + Middle | Right click |
| ✌ Index + Middle up | Scroll up |
| ✌ Index + Middle down | Scroll down |
| Q key | Quit program |

---

## Technologies Used
- Python 3.10 / 3.11  
- OpenCV  
- MediaPipe  
- PyAutoGUI  

---

## Installation

Make sure Python **3.10 or 3.11** is installed.

Install required libraries:
```bash
py -3.11 -m pip install opencv-python mediapipe pyautogui
```

---

## ▶️ How to Run

1. Open **Command Prompt**
2. Navigate to the project folder:
```bash
cd path/to/hand-gesture-virtual-mouse
```
3. Run the program:
```bash
py -3.11 virtual_mouse.py
```

---

## Learning Outcomes
- Hand landmark detection
- Gesture recognition
- Camera-to-screen coordinate mapping
- Real-time system control
- Smooth cursor movement

---

## Future Improvements
- Drag and drop gesture
- Mouse speed control
- Multi-monitor support
- Virtual keyboard integration

---

## Author
**Jatin**

 If you like this project, consider starring the repository!
