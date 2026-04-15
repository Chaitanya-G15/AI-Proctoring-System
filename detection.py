"""Proctoring detection engine for multi-user exam monitoring."""

import logging
import time
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProctorDetector:
    """Handles AI-based proctoring detection for multiple students.

    Detects violations including:
    - Looking away from screen (head pose + eye tracking)
    - Multiple people in frame
    - Prohibited objects (phones, books)
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        yaw_tolerance: float = 1.35,
        pitch_tolerance: float = 1.35,
        alert_delay: float = 1.0,
        screenshot_cooldown: float = 5.0,
        min_object_confidence: float = 0.50,
    ) -> None:
        """Initialize MediaPipe and YOLO models with detection thresholds.

        Args:
            model_path: Path to YOLO model file
            yaw_tolerance: Multiplier for yaw offset threshold (default 1.35 = 35% tolerance)
            pitch_tolerance: Multiplier for pitch offset threshold (default 1.35 = 35% tolerance)
            alert_delay: Seconds before triggering alert (default 1.0)
            screenshot_cooldown: Seconds between screenshots (default 5.0)
            min_object_confidence: Minimum confidence for YOLO detections (default 0.50 - balanced accuracy)

        Raises:
            FileNotFoundError: If YOLO model file not found
            RuntimeError: If model initialization fails
        """
        logger.info("Initializing ProctorDetector...")

        # Validate configuration
        if alert_delay < 0:
            raise ValueError(f"alert_delay must be >= 0, got {alert_delay}")
        if screenshot_cooldown < 0:
            raise ValueError(f"screenshot_cooldown must be >= 0, got {screenshot_cooldown}")
        if not (0.0 <= min_object_confidence <= 1.0):
            raise ValueError(f"min_object_confidence must be between 0 and 1, got {min_object_confidence}")

        # Thread safety for multi-user access
        self._lock = Lock()

        # Initialize MediaPipe with error handling
        try:
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                min_detection_confidence=0.5
            )
            logger.info("MediaPipe models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            raise RuntimeError(f"MediaPipe initialization failed: {e}") from e

        # Initialize YOLO with validation
        model_file = Path(model_path)
        if not model_file.exists():
            logger.error(f"YOLO model not found at {model_path}")
            raise FileNotFoundError(
                f"YOLO model not found at {model_path}. "
                f"Run 'python download_model.py' to download it."
            )

        try:
            self.yolo_model = YOLO(str(model_file))
            logger.info(f"YOLO model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError(f"YOLO model loading failed: {e}") from e

        # Detection thresholds
        self.MAX_YAW_OFFSET = 110 * yaw_tolerance
        self.MAX_PITCH_OFFSET = 140 * pitch_tolerance
        self.MIN_OBJECT_CONFIDENCE = min_object_confidence

        logger.info(f"Thresholds - Yaw: {self.MAX_YAW_OFFSET:.1f}°, Pitch: {self.MAX_PITCH_OFFSET:.1f}°")

        # 3D model points for head pose estimation (solvePnP)
        self.model_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0),
            ],
            dtype=np.float64,
        )

        # Per-student data
        self.calibrations: dict[int, dict[str, float]] = {}
        self.alert_start_times: dict[int, float] = {}
        self.last_screenshot_times: dict[int, dict[str, float]] = {}
        self.ALERT_DELAY = alert_delay
        self.SCREENSHOT_COOLDOWN = screenshot_cooldown

        logger.info("ProctorDetector initialized successfully")

    def calibrate_student(
        self, student_id: int, frames: list[np.ndarray]
    ) -> dict[str, Any]:
        """Calibrate head pose baseline for a student.

        Args:
            student_id: Unique student identifier
            frames: List of BGR image frames for calibration

        Returns:
            Calibration data with pitch, yaw, and sample count

        Raises:
            ValueError: If invalid input or no faces detected during calibration
            RuntimeError: If calibration processing fails
        """
        # Input validation
        if not isinstance(student_id, int) or student_id < 0:
            raise ValueError(f"Invalid student_id: {student_id}. Must be non-negative integer.")

        if not frames or not isinstance(frames, list):
            raise ValueError("frames must be a non-empty list of images")

        logger.info(f"Starting calibration for student {student_id} with {len(frames)} frames")

        calibration_samples = []

        for i, frame in enumerate(frames):
            try:
                # Validate frame
                if not isinstance(frame, np.ndarray):
                    logger.warning(f"Frame {i} is not a numpy array, skipping")
                    continue

                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    logger.warning(f"Frame {i} has invalid shape {frame.shape}, expected (H,W,3)")
                    continue

                height, width = frame.shape[:2]
                if height < 10 or width < 10:
                    logger.warning(f"Frame {i} too small ({width}x{height}), skipping")
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.mp_face_mesh.process(rgb)

                if not results.multi_face_landmarks:
                    logger.debug(f"No face detected in calibration frame {i}")
                    continue

                landmarks = results.multi_face_landmarks[0].landmark
                angles = self._estimate_head_pose(landmarks, width, height)

                if angles:
                    pitch, yaw, _ = angles
                    calibration_samples.append((pitch, yaw))
                    logger.debug(f"Frame {i}: pitch={pitch:.1f}, yaw={yaw:.1f}")

            except cv2.error as e:
                logger.warning(f"OpenCV error processing calibration frame {i}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing calibration frame {i}: {e}")
                continue

        if not calibration_samples:
            logger.error(f"Calibration failed for student {student_id}: No faces detected")
            raise ValueError(
                "No faces detected during calibration. Ensure:\n"
                "  1. Camera is working properly\n"
                "  2. Face is visible and well-lit\n"
                "  3. Looking directly at camera"
            )

        # Calculate calibration values
        try:
            calibrated_pitch = float(np.mean([p[0] for p in calibration_samples]))
            calibrated_yaw = float(np.mean([p[1] for p in calibration_samples]))

            with self._lock:
                self.calibrations[student_id] = {
                    "pitch": calibrated_pitch,
                    "yaw": calibrated_yaw,
                    "samples": len(calibration_samples),
                }

            logger.info(
                f"Student {student_id} calibrated: "
                f"pitch={calibrated_pitch:.1f}°, yaw={calibrated_yaw:.1f}° "
                f"({len(calibration_samples)}/{len(frames)} frames)"
            )

            return self.calibrations[student_id]

        except Exception as e:
            logger.error(f"Failed to calculate calibration for student {student_id}: {e}")
            raise RuntimeError(f"Calibration calculation failed: {e}") from e

    def process_frame(
        self, student_id: int, frame: np.ndarray
    ) -> dict[str, Any]:
        """Process a frame and detect violations.

        Args:
            student_id: Unique student identifier
            frame: BGR image frame to analyze

        Returns:
            Dict with 'violations' list (only includes those that should save screenshot),
            'details' dict, and 'all_violations' list (all detected violations)

        Raises:
            ValueError: If student not calibrated or invalid input
            RuntimeError: If frame processing fails critically
        """
        # Input validation
        if not isinstance(student_id, int) or student_id < 0:
            raise ValueError(f"Invalid student_id: {student_id}")

        if student_id not in self.calibrations:
            raise ValueError(
                f"Student {student_id} not calibrated. "
                f"Call calibrate_student() first."
            )

        if not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a numpy array")

        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(f"Invalid frame shape {frame.shape}, expected (H,W,3)")

        violations = []  # Violations to save (respects cooldown)
        all_violations = []  # All detected violations
        details: dict[str, Any] = {}

        try:
            height, width = frame.shape[:2]

            # Validate frame dimensions
            if height < 10 or width < 10:
                logger.warning(f"Frame too small ({width}x{height}), returning empty result")
                return {"violations": [], "all_violations": [], "details": {}}

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Face mesh detection with error handling
            try:
                face_mesh_results = self.mp_face_mesh.process(rgb)
                face_detection_results = self.mp_face_detection.process(rgb)
            except Exception as e:
                logger.error(f"MediaPipe processing error for student {student_id}: {e}")
                # Continue with empty results rather than failing completely
                face_mesh_results = None
                face_detection_results = None

            # Check head pose and eye tracking
            if face_mesh_results and face_mesh_results.multi_face_landmarks:
                try:
                    landmarks = face_mesh_results.multi_face_landmarks[0].landmark
                    angles = self._estimate_head_pose(landmarks, width, height)
                    eye_dir = self._get_eye_direction(landmarks)

                    if angles:
                        pitch, yaw, _ = angles
                        details["pitch"] = float(pitch)
                        details["yaw"] = float(yaw)
                        details["eye_dir"] = eye_dir

                        if self._is_looking_away(student_id, pitch, yaw) or eye_dir != "CENTER":
                            violation = self._check_alert_delay(student_id)
                            if violation:
                                all_violations.append(violation)
                                # Only save screenshot if cooldown passed
                                if self._should_save_screenshot(student_id, "not_focused"):
                                    violations.append(violation)
                        else:
                            self._reset_alert_timer(student_id)
                except Exception as e:
                    logger.warning(f"Head pose estimation error for student {student_id}: {e}")

            # Check for multiple people
            if face_detection_results and face_detection_results.detections:
                try:
                    num_faces = len(face_detection_results.detections)
                    details["num_faces"] = num_faces

                    if num_faces > 1:
                        violation = {"type": "multiple_people", "confidence": 1.0}
                        all_violations.append(violation)
                        if self._should_save_screenshot(student_id, "multiple_people"):
                            violations.append(violation)
                except Exception as e:
                    logger.warning(f"Face detection error for student {student_id}: {e}")

            # Check for prohibited objects
            try:
                object_violations = self._detect_objects(frame)
                for obj_violation in object_violations:
                    all_violations.append(obj_violation)
                    if self._should_save_screenshot(student_id, obj_violation["type"]):
                        violations.append(obj_violation)
            except Exception as e:
                logger.error(f"Object detection error for student {student_id}: {e}")

            return {
                "violations": violations,
                "all_violations": all_violations,
                "details": details
            }

        except cv2.error as e:
            logger.error(f"OpenCV error processing frame for student {student_id}: {e}")
            raise RuntimeError(f"Frame processing failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error processing frame for student {student_id}: {e}")
            raise RuntimeError(f"Frame processing failed: {e}") from e

    def _estimate_head_pose(
        self, landmarks: Any, width: int, height: int
    ) -> tuple[float, float, float] | None:
        """Estimate head pose angles using solvePnP.

        Args:
            landmarks: MediaPipe face landmarks
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Tuple of (pitch, yaw, roll) angles in degrees, or None if estimation fails
        """
        try:
            # Extract 2D facial landmarks
            image_points = np.array(
                [
                    (landmarks[1].x * width, landmarks[1].y * height),    # Nose tip
                    (landmarks[152].x * width, landmarks[152].y * height), # Chin
                    (landmarks[33].x * width, landmarks[33].y * height),   # Left eye corner
                    (landmarks[263].x * width, landmarks[263].y * height), # Right eye corner
                    (landmarks[61].x * width, landmarks[61].y * height),   # Left mouth corner
                    (landmarks[291].x * width, landmarks[291].y * height), # Right mouth corner
                ],
                dtype=np.float64,
            )

            # Validate extracted points
            if np.any(np.isnan(image_points)) or np.any(np.isinf(image_points)):
                logger.warning("Invalid landmark coordinates detected")
                return None

            # Camera intrinsic parameters
            focal_length = width
            camera_matrix = np.array(
                [[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]],
                dtype=np.float64,
            )

            # Solve PnP to get rotation vector
            success, rotation_vector, _ = cv2.solvePnP(
                self.model_points,
                image_points,
                camera_matrix,
                np.zeros((4, 1)),
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                logger.debug("solvePnP failed to converge")
                return None

            # Convert rotation vector to rotation matrix
            rmat, _ = cv2.Rodrigues(rotation_vector)

            # Extract Euler angles from rotation matrix
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            # Validate angles
            if np.any(np.isnan(angles)) or np.any(np.isinf(angles)):
                logger.warning("Invalid angles computed from rotation matrix")
                return None

            return angles  # pitch, yaw, roll

        except cv2.error as e:
            logger.warning(f"OpenCV error in head pose estimation: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error in head pose estimation: {e}")
            return None

    def _is_looking_away(self, student_id: int, pitch: float, yaw: float) -> bool:
        """Check if student is looking away from screen."""
        calibration = self.calibrations[student_id]
        pitch_offset = abs(pitch - calibration["pitch"])
        yaw_offset = abs(yaw - calibration["yaw"])
        return pitch_offset > self.MAX_PITCH_OFFSET or yaw_offset > self.MAX_YAW_OFFSET

    def _get_eye_direction(self, landmarks: Any) -> str:
        """Get eye direction based on iris position.

        Args:
            landmarks: MediaPipe face landmarks

        Returns:
            'LEFT', 'RIGHT', or 'CENTER'
        """
        try:
            left = landmarks[33]   # Left eye corner
            right = landmarks[133] # Right eye corner
            iris = landmarks[468]  # Left iris center

            # Validate landmark coordinates
            if any(hasattr(lm, 'x') and (np.isnan(lm.x) or np.isnan(lm.y))
                   for lm in [left, right, iris]):
                logger.debug("Invalid eye landmark coordinates")
                return "CENTER"

            eye_width = right.x - left.x

            # Guard against division by zero or negative width
            if eye_width <= 1e-6:
                logger.debug(f"Invalid eye width: {eye_width}")
                return "CENTER"

            iris_pos = (iris.x - left.x) / eye_width

            # Thresholds for left/right detection
            if iris_pos < 0.35:
                return "LEFT"
            if iris_pos > 0.65:
                return "RIGHT"
            return "CENTER"

        except (AttributeError, IndexError, TypeError) as e:
            logger.warning(f"Error getting eye direction: {e}")
            return "CENTER"

    def _check_alert_delay(self, student_id: int) -> dict[str, Any] | None:
        """Check if alert delay has passed before triggering violation.

        Args:
            student_id: Student ID

        Returns:
            Violation dict if alert delay passed, None otherwise
        """
        with self._lock:
            if student_id not in self.alert_start_times:
                self.alert_start_times[student_id] = time.time()
                return None

            if time.time() - self.alert_start_times[student_id] > self.ALERT_DELAY:
                return {"type": "not_focused", "confidence": 1.0}

        return None

    def _reset_alert_timer(self, student_id: int) -> None:
        """Reset alert timer when student is focused.

        Args:
            student_id: Student ID
        """
        with self._lock:
            if student_id in self.alert_start_times:
                del self.alert_start_times[student_id]

    def _should_save_screenshot(self, student_id: int, violation_type: str) -> bool:
        """Check if enough time has passed to save another screenshot.

        Args:
            student_id: Student ID
            violation_type: Type of violation

        Returns:
            True if screenshot should be saved (cooldown expired)
        """
        with self._lock:
            if student_id not in self.last_screenshot_times:
                self.last_screenshot_times[student_id] = {}

            last_time = self.last_screenshot_times[student_id].get(violation_type, 0)
            current_time = time.time()

            if current_time - last_time >= self.SCREENSHOT_COOLDOWN:
                self.last_screenshot_times[student_id][violation_type] = current_time
                return True

            return False

    def _detect_objects(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Detect prohibited objects using YOLO.

        Args:
            frame: BGR image frame

        Returns:
            List of violation dictionaries with type and confidence
        """
        violations = []

        try:
            # Run YOLO inference
            yolo_results = self.yolo_model(frame, stream=True, verbose=False)

            for result in yolo_results:
                if not hasattr(result, 'boxes') or result.boxes is None:
                    continue

                for box in result.boxes:
                    try:
                        # Extract class and confidence
                        cls_id = int(box.cls[0])
                        cls_name = result.names[cls_id]
                        confidence = float(box.conf[0])

                        # Filter by confidence threshold
                        if confidence < self.MIN_OBJECT_CONFIDENCE:
                            logger.debug(
                                f"Ignoring {cls_name} detection "
                                f"(confidence {confidence:.2f} < {self.MIN_OBJECT_CONFIDENCE})"
                            )
                            continue

                        # Check if it's a prohibited object
                        if cls_name in ["cell phone", "book"]:
                            violation_type = "phone" if cls_name == "cell phone" else "book"
                            violations.append({
                                "type": violation_type,
                                "confidence": confidence
                            })
                            logger.debug(
                                f"Detected {violation_type} with confidence {confidence:.2f}"
                            )

                    except (IndexError, KeyError, ValueError) as e:
                        logger.warning(f"Error processing YOLO detection box: {e}")
                        continue

        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            # Return empty list rather than crashing
            return []

        return violations

    def cleanup(self) -> None:
        """Cleanup resources and close models.

        Call this when done with the detector to free resources.
        """
        try:
            logger.info("Cleaning up ProctorDetector resources...")

            if hasattr(self, 'mp_face_mesh') and self.mp_face_mesh:
                self.mp_face_mesh.close()

            if hasattr(self, 'mp_face_detection') and self.mp_face_detection:
                self.mp_face_detection.close()

            logger.info("ProctorDetector cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False
