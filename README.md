# ML Laterals Spider – Task Repository

This repository contains implementations for **Computer Vision** tasks completed as part of the **ML Laterals Spider Program**.

---

## 📁 Repository Structure

```
├── OpenCV/                       # Computer Vision Tasks
│   ├── Task_1/                   # Object Boundary Tracking
│   └── Task_2/                   # Eye Tracking Systems
```
## Projects Overview


### OpenCV – Task 2: Eye Tracking Systems

---

#### A. Dlib-Based Eye Tracker

Advanced eye tracking system with gaze detection and cursor control.

##### Features
- Eye Aspect Ratio (EAR) for blink detection
- Pupil detection using thresholding and morphological operations
- Gaze direction estimation (LEFT, RIGHT, UP, DOWN, CENTER)
- Smooth cursor movement with interpolation
- Blink-based interactions:
  - Double Blink: Left Click
  - Triple Blink: Right Click
  - Gaze Hold: Click after sustained center gaze

##### Key Constants
- EAR Threshold: 0.22
- Consecutive Blink Frames: 3
- Cursor Smoothing Factor: 0.3
- Gaze Hold Time: 1.2 seconds

##### Technologies
OpenCV, Dlib, imutils, SciPy, PyAutoGUI
Dlib model which i used is already there in Task_2/Bonus/Model directory

---

#### B. MediaPipe-Based Eye Tracker

Lightweight and faster alternative using MediaPipe Face Mesh.

##### Features
- 468-landmark facial mesh detection
- Iris landmark tracking (landmarks 468–475)
- Smooth cursor control with exponential smoothing
- Blink detection and counting
- Click actions via blinks and gaze hold

##### Advantages
- No external model files required
- Faster processing
- More robust facial landmark detection

##### Technologies
OpenCV, MediaPipe, NumPy, SciPy, PyAutoGUI

---

## Getting Started

### Prerequisites
- Python 3.9+
- Conda (recommended)

---

## Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/ML_Laterals_Spider.git
cd ML_Laterals_Spider
```

---

### Environment Setup

#### OpenCV Task 1
```bash
cd OpenCV/Task_1
conda env create -f webcam.yml
conda activate Object_tracking
```

#### OpenCV Task 2 – Dlib
```bash
cd OpenCV/Task_2/Bonus
conda env create -f track_Bonus.yml
conda activate Bonus_eye_tracker
```

#### OpenCV Task 2 – MediaPipe
```bash
cd OpenCV/Task_2/Mediapipe
conda env create -f track_Mediapipe.yml
conda activate mediapipe_eye_tracker
```

---

## Running the Projects
`

#### Object Tracking
```bash
cd OpenCV/Task_1
python Webcam_modular.py
# Press ESC to exit
```

#### Eye Tracking (Dlib)
```bash
cd OpenCV/Task_2/Bonus
python track_Bonus.py
# Press 'q' to quit
```

#### Eye Tracking (MediaPipe)
```bash
cd OpenCV/Task_2/Mediapipe
python track_Mediapipe.py
# Press 'q' to quit
```


---

## Notes

### Eye Tracking Requirements
```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

- Ensure adequate lighting
- Camera should have a clear view of your face

---

---

## Contributing
Feel free to open issues or submit pull requests for improvements.

---

## Contact
For questions or collaborations, reach out via GitHub issues.

---
