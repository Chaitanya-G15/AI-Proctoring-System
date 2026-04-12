# Face Detection Only - AI Proctoring System

A lightweight version of the AI Proctoring System that uses **only MediaPipe** for face detection, without YOLO or PyTorch dependencies.

> **Note:** There's also a hybrid version in `cv2-yolo-hybrid/` that uses OpenCV Haar Cascades + YOLO for phone/book detection.

## Features

### ✅ What's Included
- 👁️ **Eye Tracking (Iris-based)** - Detects if the user is looking away
- 🧭 **Head Pose Estimation** - Tracks head movement (left/right/up/down)
- 👤 **Face Detection** - Ensures candidate presence
- 👥 **Multiple Person Detection** - Alerts when more than one person is detected
- 🔔 **Alert System** - Visual and audio alerts for violations
- 📸 **Screenshot Capture** - Saves timestamped screenshots to `violations/` folder
- 📊 **Session Summary** - Shows percentage of time looking away and multiple people detected

### ❌ What's NOT Included (compared to full version)
- Phone detection (requires YOLO)
- Book detection (requires YOLO)

## Advantages

- **Lightweight** - No PyTorch or YOLO dependencies (~6GB smaller)
- **Faster** - Runs quicker without object detection
- **Easier Setup** - Simpler installation process
- **Lower Requirements** - Works on less powerful hardware

## Installation

1. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python main.py
```

### Steps:
1. Face the camera for **calibration (3 seconds)**
2. System starts monitoring automatically
3. Press **'q'** to exit and see session summary

## Violations Detected

1. **Not Focused** - User looking away from screen (head pose or eye direction)
2. **Multiple People** - More than one person detected in frame

## Screenshot Storage

All violations are automatically saved to the `violations/` folder with:
- Violation type label
- Timestamp overlay (bottom-right corner)
- Unix timestamp in filename

Example: `violations/not_focused_1775921570.jpg`

## Platform Notes

- **Windows** - Full support including audio alerts
- **Linux/Mac** - Works but audio alerts are disabled (winsound not available)

## Requirements

- Working webcam
- Good lighting conditions
- Python 3.8+

## Comparison with Full Version

| Feature | This Version | Full Version |
|---------|--------------|--------------|
| Face Detection | ✅ | ✅ |
| Head Pose Tracking | ✅ | ✅ |
| Eye Tracking | ✅ | ✅ |
| Multiple Person Detection | ✅ | ✅ |
| Phone Detection | ❌ | ✅ |
| Book Detection | ❌ | ✅ |
| Installation Size | ~500MB | ~6.5GB |
| Speed | Faster | Slower |

## When to Use This Version

Choose this lightweight version if:
- You don't need phone/book detection
- You want faster performance
- You have limited disk space or bandwidth
- You're running on lower-end hardware
- You want simpler installation and maintenance

For full object detection capabilities, use the main version in the parent directory.
