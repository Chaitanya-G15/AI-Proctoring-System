# import cv2
# import mediapipe as mp
# import numpy as np
# from ultralytics import YOLO
# import time

# # Initialize MediaPipe and YOLO
# mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
# mp_face_detection = mp.solutions.face_detection.FaceDetection()
# yolo_model = YOLO('yolov8n.pt')

# # Tracking variables
# away_count = 0
# phone_detected_count = 0
# unauthorized_person_detected_count = 0
# total_frames = 0

# # Calibration values
# calibrated_pitch = 0
# calibrated_yaw = 0

# # Thresholds (increased by 20%)
# MAX_YAW_OFFSET = 110 * 1.35  # Increased by 35%
# MAX_PITCH_OFFSET = 140 * 1.35  # Increased by 35%

# # 3D Model points for head pose estimation
# model_points = np.array([
#     (0.0, 0.0, 0.0),
#     (0.0, -330.0, -65.0),
#     (-225.0, 170.0, -135.0),
#     (225.0, 170.0, -135.0),
#     (-150.0, -150.0, -125.0),
#     (150.0, -150.0, -125.0)
# ], dtype=np.float64)

# # Estimate head pose
# def estimate_head_pose(landmarks, width, height):
#     image_points = np.array([
#         (landmarks[1].x * width, landmarks[1].y * height),
#         (landmarks[152].x * width, landmarks[152].y * height),
#         (landmarks[33].x * width, landmarks[33].y * height),
#         (landmarks[263].x * width, landmarks[263].y * height),
#         (landmarks[61].x * width, landmarks[61].y * height),
#         (landmarks[291].x * width, landmarks[291].y * height)
#     ], dtype=np.float64)

#     focal_length = width
#     camera_matrix = np.array([
#         [focal_length, 0, width / 2],
#         [0, focal_length, height / 2],
#         [0, 0, 1]
#     ], dtype=np.float64)

#     success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, np.zeros((4, 1)))
#     if not success:
#         return None

#     rmat, _ = cv2.Rodrigues(rotation_vector)
#     angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
#     return angles  # pitch, yaw, roll

# # Check if user is looking away
# def is_looking_away(pitch, yaw):
#     pitch_offset = abs(pitch - calibrated_pitch)
#     yaw_offset = abs(yaw - calibrated_yaw)
#     return pitch_offset > MAX_PITCH_OFFSET or yaw_offset > MAX_YAW_OFFSET

# # Detect multiple faces
# def detect_multiple_faces(detections):
#     return len(detections) > 1

# # Start capturing
# cap = cv2.VideoCapture(0)

# # === PRE-CALIBRATION TIMER AND MESSAGE ===
# for i in range(5, 0, -1):
#     ret, frame = cap.read()
#     if not ret:
#         break
#     cv2.putText(frame, "Please face the camera directly for calibration.", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#     cv2.putText(frame, f"Calibration starts in: {i}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#     cv2.imshow("Proctoring System", frame)
#     cv2.waitKey(1000)

# # === CALIBRATION PHASE ===
# calibration_frames = []

# start_time = time.time()
# while time.time() - start_time < 3:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     height, width, _ = frame.shape
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     face_mesh_results = mp_face_mesh.process(rgb_frame)
#     if face_mesh_results.multi_face_landmarks:
#         landmarks = face_mesh_results.multi_face_landmarks[0].landmark
#         angles = estimate_head_pose(landmarks, width, height)
#         if angles:
#             pitch, yaw, _ = angles
#             calibration_frames.append((pitch, yaw))

#     seconds_left = 3 - int(time.time() - start_time)
#     cv2.putText(frame, "Calibrating... Keep your face straight!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#     cv2.putText(frame, f"Seconds left: {seconds_left}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#     cv2.imshow("Proctoring System", frame)
#     cv2.waitKey(1)

# if calibration_frames:
#     calibrated_pitch = np.mean([p[0] for p in calibration_frames])
#     calibrated_yaw = np.mean([p[1] for p in calibration_frames])

# # === MAIN MONITORING LOOP ===
# start_session_time = time.time()

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     total_frames += 1
#     height, width, _ = frame.shape
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     face_mesh_results = mp_face_mesh.process(rgb_frame)
#     face_detection_results = mp_face_detection.process(rgb_frame)

#     phone_detected = False

#     if face_mesh_results.multi_face_landmarks:
#         landmarks = face_mesh_results.multi_face_landmarks[0].landmark
#         angles = estimate_head_pose(landmarks, width, height)

#         if angles:
#             pitch, yaw, roll = angles

#             if is_looking_away(pitch, yaw):
#                 away_count += 1
#                 cv2.putText(frame, "LOOKING AWAY!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#             cv2.putText(frame, f"Pitch: {pitch:.1f}", (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#             cv2.putText(frame, f"Yaw: {yaw:.1f}", (width - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#     if face_detection_results.detections and detect_multiple_faces(face_detection_results.detections):
#         unauthorized_person_detected_count += 1
#         cv2.putText(frame, "MULTIPLE PEOPLE DETECTED!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     yolo_results = yolo_model(frame, stream=True, verbose=False)
#     for result in yolo_results:
#         for box in result.boxes:
#             cls = result.names[int(box.cls[0])]
#             x1, y1, x2, y2 = map(int, box.xyxy[0])

#             if cls in ["cell phone", "book"]:
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 phone_detected = cls == "cell phone"

#     if phone_detected:
#         phone_detected_count += 1
#         cv2.putText(frame, "PHONE DETECTED!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     cv2.imshow("Proctoring System", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# # === SESSION SUMMARY ===
# duration = int(time.time() - start_session_time)
# print(f"\n=== SESSION SUMMARY ===")
# print(f"Total Duration: {duration} seconds")
# print(f"Looking Away: {away_count / total_frames * 100:.2f}%")
# print(f"Phone Detection: {phone_detected_count / total_frames * 100:.2f}%")
# print(f"Unauthorized Person Detection: {unauthorized_person_detected_count / total_frames * 100:.2f}%")


import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import time
import os
import winsound   # For sound alert (Windows)

# ================= INITIAL SETUP =================
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
mp_face_detection = mp.solutions.face_detection.FaceDetection()
yolo_model = YOLO('yolov8n.pt')

# Create folder for violations
if not os.path.exists("violations"):
    os.makedirs("violations")

# Tracking variables
away_count = 0
phone_detected_count = 0
unauthorized_person_detected_count = 0
total_frames = 0

# Alert control
alert_start_time = None
ALERT_DELAY = 2  # seconds
last_beep_time = 0

# Calibration
calibrated_pitch = 0
calibrated_yaw = 0

# Thresholds
MAX_YAW_OFFSET = 110 * 1.35
MAX_PITCH_OFFSET = 140 * 1.35

# 3D model points
model_points = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)

# ================= FUNCTIONS =================

def save_violation(frame, label):
    timestamp = int(time.time())
    filename = f"violations/{label}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)

def show_alert(frame, message, y_pos=50):
    global last_beep_time

    # Red alert box
    cv2.rectangle(frame, (20, y_pos - 30), (600, y_pos + 10), (0, 0, 255), -1)

    # Text
    cv2.putText(frame, message, (30, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Beep every 2 sec (not continuous)
    if time.time() - last_beep_time > 2:
        winsound.Beep(1000, 300)
        last_beep_time = time.time()

def estimate_head_pose(landmarks, width, height):
    image_points = np.array([
        (landmarks[1].x * width, landmarks[1].y * height),
        (landmarks[152].x * width, landmarks[152].y * height),
        (landmarks[33].x * width, landmarks[33].y * height),
        (landmarks[263].x * width, landmarks[263].y * height),
        (landmarks[61].x * width, landmarks[61].y * height),
        (landmarks[291].x * width, landmarks[291].y * height)
    ], dtype=np.float64)

    focal_length = width
    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, np.zeros((4, 1)))
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles

def is_looking_away(pitch, yaw):
    pitch_offset = abs(pitch - calibrated_pitch)
    yaw_offset = abs(yaw - calibrated_yaw)
    return pitch_offset > MAX_PITCH_OFFSET or yaw_offset > MAX_YAW_OFFSET

def detect_multiple_faces(detections):
    return len(detections) > 1

def get_eye_direction(landmarks):
    left = landmarks[33]
    right = landmarks[133]
    iris = landmarks[468]

    eye_width = right.x - left.x
    iris_pos = (iris.x - left.x) / eye_width

    if iris_pos < 0.35:
        return "LEFT"
    elif iris_pos > 0.65:
        return "RIGHT"
    else:
        return "CENTER"

# ================= START CAMERA =================
cap = cv2.VideoCapture(0)

# ================= CALIBRATION =================
print("Look straight at camera for calibration...")
calibration_frames = []
start_time = time.time()

while time.time() - start_time < 3:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = mp_face_mesh.process(rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        angles = estimate_head_pose(landmarks, width, height)
        if angles:
            calibration_frames.append(angles[:2])

    cv2.putText(frame, "Calibrating...", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.imshow("Calibration", frame)
    cv2.waitKey(1)

if calibration_frames:
    calibrated_pitch = np.mean([p[0] for p in calibration_frames])
    calibrated_yaw = np.mean([p[1] for p in calibration_frames])

print("Calibration done!")

# ================= MAIN LOOP =================
start_session_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1
    height, width, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_mesh_results = mp_face_mesh.process(rgb)
    face_detection_results = mp_face_detection.process(rgb)

    phone_detected = False

    # ===== FACE + HEAD + EYE =====
    if face_mesh_results.multi_face_landmarks:
        landmarks = face_mesh_results.multi_face_landmarks[0].landmark

        angles = estimate_head_pose(landmarks, width, height)
        eye_dir = get_eye_direction(landmarks)

        if angles:
            pitch, yaw, _ = angles

            if is_looking_away(pitch, yaw) or eye_dir != "CENTER":
                if alert_start_time is None:
                    alert_start_time = time.time()
                elif time.time() - alert_start_time > ALERT_DELAY:
                    away_count += 1
                    show_alert(frame, "⚠️ NOT FOCUSED!", 50)
                    save_violation(frame, "not_focused")
            else:
                alert_start_time = None

    # ===== MULTIPLE FACE =====
    if face_detection_results.detections and detect_multiple_faces(face_detection_results.detections):
        unauthorized_person_detected_count += 1
        show_alert(frame, "⚠️ MULTIPLE PEOPLE DETECTED!", 100)
        save_violation(frame, "multiple_people")

    # ===== YOLO OBJECT DETECTION =====
    results = yolo_model(frame, stream=True, verbose=False)
    for r in results:
        for box in r.boxes:
            cls = r.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls in ["cell phone", "book"]:
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,0,255),2)
                phone_detected = cls == "cell phone"

    if phone_detected:
        phone_detected_count += 1
        show_alert(frame, "⚠️ PHONE DETECTED!", 150)
        save_violation(frame, "phone")

    # ===== STATUS INDICATOR =====
    status = "FOCUSED"
    color = (0, 255, 0)

    if alert_start_time:
        status = "WARNING"
        color = (0, 0, 255)

    cv2.putText(frame, f"STATUS: {status}", (width - 220, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ===== DISPLAY =====
    cv2.imshow("Proctoring System", frame)

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
print(f"Phone: {phone_detected_count/total_frames*100:.2f}%")
print(f"Multiple People: {unauthorized_person_detected_count/total_frames*100:.2f}%")