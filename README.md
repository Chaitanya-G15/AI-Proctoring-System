# 🎓 AI Proctoring System (Exam Cheating Detection)

An advanced **AI-powered exam proctoring system** built using Python, OpenCV, MediaPipe, and YOLOv8.
This system monitors users in real-time during online exams and detects suspicious activities using computer vision.

---

## 🚀 Features

### 🧠 Intelligent Monitoring

* 👁️ **Eye Tracking (Iris-based)** – Detects if the user is looking away from the screen
* 🧭 **Head Pose Estimation** – Tracks head movement (left/right/up/down)
* 👤 **Face Detection** – Ensures candidate presence

### 🚨 Cheating Detection

* 📱 **Phone Detection** using YOLOv8
* 👥 **Multiple Person Detection**
* 📖 Detection of suspicious objects (e.g., books)

### 🔔 Alert System

* 🔴 Real-time **on-screen warning alerts**
* 🔊 **Audio alert (beep sound)** for violations
* 🟢 Live **status indicator (Focused / Warning)**

### 📸 Evidence & Logging

* Automatic **screenshot capture** on violations
* Stores images in `violations/` folder
* 📊 Generates **session summary**:

  * % Looking away
  * % Phone usage
  * % Multiple person detection

---

## 🛠️ Tech Stack

* Python
* OpenCV
* MediaPipe
* YOLOv8 (Ultralytics)
* NumPy

---

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/AI-Proctoring-System.git
cd AI-Proctoring-System
```

2. Create virtual environment:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the application:

```bash
python main.py
```

### 🧪 Steps:

1. Face the camera for **calibration (3 seconds)**
2. System starts monitoring automatically
3. Press **'q'** to exit

---

## 📁 Project Structure

```
AI-Proctoring-System/
│── main.py
│── requirements.txt
│── violations/        # Auto-generated screenshots
│── README.md
│── .gitignore
```

---

## ⚠️ Limitations

* Performance depends on lighting conditions
* Requires a working webcam
* `winsound` alerts work only on Windows

---

## 🙌 Credits

* Original project by: AaravMehta-07
* Enhanced and upgraded by: *Your Name*

Enhancements include:

* Eye tracking (iris-based)
* Smart alert system (visual + audio)
* Screenshot logging
* Improved detection accuracy

---

## 🎯 Future Improvements

* Face recognition (authorized candidate only)
* Voice detection (detect talking)
* Web-based dashboard (Django/Flask)
* AI-based cheating score system

---

## ⭐ Conclusion

This project demonstrates how **computer vision and AI** can be used to build a **real-time automated proctoring system**, making online exams more secure and reliable.

---
