import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import winsound   # For sound alert (Windows)
from datetime import datetime

# ================= INITIAL SETUP =================
# Load OpenCV's pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load YOLO model
yolo_model = YOLO('yolov8n.pt')

# Create folder for violations
if not os.path.exists("violations"):
    os.makedirs("violations")

# Tracking variables
away_count = 0
phone_detected_count = 0
earphone_detected_count = 0
watch_detected_count = 0
unauthorized_person_detected_count = 0
total_frames = 0

# Alert control
last_beep_time = 0
ALERT_DELAY = 2  # seconds

# Screenshot cooldown system (prevent data burst)
SCREENSHOT_COOLDOWN = 10  # seconds between saving same violation type
last_screenshot_time = {
    "no_face": 0,
    "not_focused": 0,
    "multiple_people": 0,
    "phone": 0,
    "earphones": 0,
    "watch": 0
}

# Face tracking for "looking away" detection
no_face_frames = 0
no_eyes_frames = 0
NO_FACE_THRESHOLD = 15  # frames before triggering alert
NO_EYES_THRESHOLD = 10  # frames before triggering alert

# ================= FUNCTIONS =================

def save_violation(frame, label):
    global last_screenshot_time

    # Check cooldown - prevent saving multiple screenshots of same violation too quickly
    current_time = time.time()
    if current_time - last_screenshot_time[label] < SCREENSHOT_COOLDOWN:
        return  # Skip saving, cooldown period not elapsed

    # Update last screenshot time for this violation type
    last_screenshot_time[label] = current_time

    # Create a copy to avoid modifying the display frame
    save_frame = frame.copy()

    # Get formatted timestamp
    timestamp = int(time.time())
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add timestamp text to the image (bottom-right corner with background)
    text = f"Time: {timestamp_str}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    # Get text size for background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Position at bottom-right corner
    height, width = save_frame.shape[:2]
    x = width - text_width - 20
    y = height - 20

    # Draw black background rectangle
    cv2.rectangle(save_frame, (x - 10, y - text_height - 10),
                  (x + text_width + 10, y + baseline + 10), (0, 0, 0), -1)

    # Draw timestamp text in white
    cv2.putText(save_frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)

    # Save the frame with timestamp
    filename = f"violations/{label}_{timestamp}.jpg"
    cv2.imwrite(filename, save_frame)
    print(f"📸 Screenshot saved: {filename}")

def show_alert(frame, message, y_pos=50):
    global last_beep_time

    # Red alert box
    cv2.rectangle(frame, (20, y_pos - 30), (600, y_pos + 10), (0, 0, 255), -1)

    # Text
    cv2.putText(frame, message, (30, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Beep every 2 sec (not continuous)
    if time.time() - last_beep_time > 2:
        try:
            winsound.Beep(1000, 300)
        except:
            pass  # Skip audio on non-Windows platforms
        last_beep_time = time.time()

def detect_faces_and_eyes(gray_frame, frame):
    """Detect faces and eyes using OpenCV Haar Cascades"""
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    eyes_detected = False

    # Draw rectangles around faces and detect eyes
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Region of interest for eye detection (upper half of face)
        roi_gray = gray_frame[y:y+int(h/2), x:x+w]
        roi_color = frame[y:y+int(h/2), x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)

        if len(eyes) >= 2:
            eyes_detected = True
            for (ex, ey, ew, eh) in eyes[:2]:  # Draw only first 2 eyes
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 1)

    return len(faces), eyes_detected

# ================= START CAMERA =================
cap = cv2.VideoCapture(0)

print("Starting CV2 + YOLO Hybrid Proctoring System...")
print("Press 'q' to quit")

# ================= MAIN LOOP =================
start_session_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1
    height, width, _ = frame.shape

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ===== FACE DETECTION (CV2) =====
    num_faces, eyes_detected = detect_faces_and_eyes(gray, frame)

    # Track "looking away" - no face detected
    if num_faces == 0:
        no_face_frames += 1
        if no_face_frames > NO_FACE_THRESHOLD:
            away_count += 1
            show_alert(frame, "⚠️ FACE NOT DETECTED!", 50)
            if no_face_frames == NO_FACE_THRESHOLD + 1:  # Save once when threshold crossed
                save_violation(frame, "no_face")
    else:
        no_face_frames = 0

    # Track "looking away" - no eyes detected (looking down/away)
    if num_faces > 0 and not eyes_detected:
        no_eyes_frames += 1
        if no_eyes_frames > NO_EYES_THRESHOLD:
            away_count += 1
            show_alert(frame, "⚠️ NOT FOCUSED!", 50)
            if no_eyes_frames == NO_EYES_THRESHOLD + 1:  # Save once when threshold crossed
                save_violation(frame, "not_focused")
    else:
        no_eyes_frames = 0

    # ===== MULTIPLE FACE DETECTION =====
    if num_faces > 1:
        unauthorized_person_detected_count += 1
        show_alert(frame, "⚠️ MULTIPLE PEOPLE DETECTED!", 100)
        save_violation(frame, "multiple_people")

    # ===== YOLO OBJECT DETECTION (Phone/Book/Earphones/Watch) =====
    phone_detected = False
    earphone_detected = False
    watch_detected = False
    results = yolo_model(frame, stream=True, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = r.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Detect prohibited items
            if cls in ["cell phone", "book", "earphones", "headphones", "earbuds", "watch"]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, cls.upper(), (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                if cls == "cell phone":
                    phone_detected = True
                elif cls in ["earphones", "headphones", "earbuds"]:
                    earphone_detected = True
                elif cls == "watch":
                    watch_detected = True

    if phone_detected:
        phone_detected_count += 1
        show_alert(frame, "⚠️ PHONE DETECTED!", 150)
        save_violation(frame, "phone")

    if earphone_detected:
        earphone_detected_count += 1
        show_alert(frame, "⚠️ EARPHONES DETECTED!", 200)
        save_violation(frame, "earphones")

    if watch_detected:
        watch_detected_count += 1
        show_alert(frame, "⚠️ SMART WATCH DETECTED!", 250)
        save_violation(frame, "watch")

    # ===== STATUS INDICATOR =====
    status = "FOCUSED"
    color = (0, 255, 0)

    if no_face_frames > NO_FACE_THRESHOLD or no_eyes_frames > NO_EYES_THRESHOLD:
        status = "WARNING"
        color = (0, 0, 255)

    cv2.putText(frame, f"STATUS: {status}", (width - 220, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Face count indicator
    cv2.putText(frame, f"Faces: {num_faces}", (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ===== DISPLAY =====
    cv2.imshow("Proctoring System (CV2 + YOLO)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()

# ================= REPORT =================
duration = int(time.time() - start_session_time)

print("\n=== SESSION SUMMARY ===")
print(f"Duration: {duration} sec")
print(f"Looking Away: {away_count/total_frames*100:.2f}%")
print(f"Phone Detection: {phone_detected_count/total_frames*100:.2f}%")
print(f"Earphone Detection: {earphone_detected_count/total_frames*100:.2f}%")
print(f"Watch Detection: {watch_detected_count/total_frames*100:.2f}%")
print(f"Multiple People: {unauthorized_person_detected_count/total_frames*100:.2f}%")
