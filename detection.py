import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import time

class ProctorDetector:
    """
    Proctoring detector that can handle multiple students simultaneously.
    Extracted from main.py with multi-user support.
    """

    def __init__(self):
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection()

        # Initialize YOLO
        self.yolo_model = YOLO('models/yolov8n.pt')

        # Detection thresholds
        self.MAX_YAW_OFFSET = 110 * 1.35  # 148.5 degrees
        self.MAX_PITCH_OFFSET = 140 * 1.35  # 189 degrees

        # 3D model points for head pose estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float64)

        # Store calibration data per student
        self.calibrations = {}  # student_id -> {'pitch': float, 'yaw': float}

        # Alert tracking per student
        self.alert_start_times = {}  # student_id -> timestamp
        self.ALERT_DELAY = 2  # seconds before triggering alert

    def calibrate_student(self, student_id, frames):
        """
        Calibrate head pose for a specific student.

        Args:
            student_id: Unique student identifier
            frames: List of CV2 image frames (BGR format)

        Returns:
            dict: Calibration data {'pitch': float, 'yaw': float, 'samples': int}
        """
        calibration_samples = []

        for frame in frames:
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.mp_face_mesh.process(rgb)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                angles = self._estimate_head_pose(landmarks, width, height)
                if angles:
                    pitch, yaw, _ = angles
                    calibration_samples.append((pitch, yaw))

        if calibration_samples:
            calibrated_pitch = np.mean([p[0] for p in calibration_samples])
            calibrated_yaw = np.mean([p[1] for p in calibration_samples])

            self.calibrations[student_id] = {
                'pitch': calibrated_pitch,
                'yaw': calibrated_yaw,
                'samples': len(calibration_samples)
            }

            return self.calibrations[student_id]
        else:
            raise ValueError("No faces detected during calibration")

    def process_frame(self, student_id, frame):
        """
        Process a single frame and detect violations.

        Args:
            student_id: Unique student identifier
            frame: CV2 image frame (BGR format)

        Returns:
            dict: {
                'violations': [{'type': str, 'confidence': float}],
                'details': {'pitch': float, 'yaw': float, 'eye_dir': str, ...}
            }
        """
        if student_id not in self.calibrations:
            raise ValueError(f"Student {student_id} not calibrated")

        violations = []
        details = {}

        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face mesh detection (head pose + eye tracking)
        face_mesh_results = self.mp_face_mesh.process(rgb)

        # Multi-face detection
        face_detection_results = self.mp_face_detection.process(rgb)

        # Head pose and eye tracking
        if face_mesh_results.multi_face_landmarks:
            landmarks = face_mesh_results.multi_face_landmarks[0].landmark

            angles = self._estimate_head_pose(landmarks, width, height)
            eye_dir = self._get_eye_direction(landmarks)

            if angles:
                pitch, yaw, roll = angles
                details['pitch'] = float(pitch)
                details['yaw'] = float(yaw)
                details['eye_dir'] = eye_dir

                # Check if looking away
                if self._is_looking_away(student_id, pitch, yaw) or eye_dir != "CENTER":
                    # Use alert delay to avoid false positives
                    if student_id not in self.alert_start_times:
                        self.alert_start_times[student_id] = time.time()
                    elif time.time() - self.alert_start_times[student_id] > self.ALERT_DELAY:
                        violations.append({
                            'type': 'not_focused',
                            'confidence': 1.0
                        })
                else:
                    # Reset alert timer
                    if student_id in self.alert_start_times:
                        del self.alert_start_times[student_id]

        # Multiple people detection
        if face_detection_results.detections and self._detect_multiple_faces(face_detection_results.detections):
            violations.append({
                'type': 'multiple_people',
                'confidence': 1.0
            })
            details['num_faces'] = len(face_detection_results.detections)

        # Object detection (phone, book)
        phone_detected = False
        yolo_results = self.yolo_model(frame, stream=True, verbose=False)
        for r in yolo_results:
            for box in r.boxes:
                cls = r.names[int(box.cls[0])]
                confidence = float(box.conf[0])

                if cls in ["cell phone", "book"]:
                    if cls == "cell phone":
                        phone_detected = True
                        violations.append({
                            'type': 'phone',
                            'confidence': confidence
                        })
                    details[f'{cls}_detected'] = True

        return {
            'violations': violations,
            'details': details
        }

    def _estimate_head_pose(self, landmarks, width, height):
        """
        Estimate head pose using solvePnP.
        Extracted from main.py lines 277-300.
        """
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

        success, rotation_vector, _ = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            np.zeros((4, 1))
        )

        if not success:
            return None

        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return angles  # pitch, yaw, roll

    def _is_looking_away(self, student_id, pitch, yaw):
        """
        Check if student is looking away based on calibrated values.
        Extracted from main.py lines 302-305.
        """
        calibration = self.calibrations[student_id]
        pitch_offset = abs(pitch - calibration['pitch'])
        yaw_offset = abs(yaw - calibration['yaw'])
        return pitch_offset > self.MAX_PITCH_OFFSET or yaw_offset > self.MAX_YAW_OFFSET

    def _detect_multiple_faces(self, detections):
        """
        Check if multiple faces are detected.
        Extracted from main.py lines 307-308.
        """
        return len(detections) > 1

    def _get_eye_direction(self, landmarks):
        """
        Get eye direction based on iris position.
        Extracted from main.py lines 310-323.
        """
        left = landmarks[33]
        right = landmarks[133]
        iris = landmarks[468]

        eye_width = right.x - left.x
        if eye_width == 0:
            return "CENTER"

        iris_pos = (iris.x - left.x) / eye_width

        if iris_pos < 0.35:
            return "LEFT"
        elif iris_pos > 0.65:
            return "RIGHT"
        else:
            return "CENTER"
