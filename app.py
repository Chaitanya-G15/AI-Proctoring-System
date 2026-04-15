"""Flask web server for multi-user AI proctoring system."""

import base64
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, render_template, request, send_from_directory

from database import Database
from detection import ProctorDetector

app = Flask(__name__)

# Disable caching for development/demo
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# Global state
detector = ProctorDetector()
db = Database()
exam_started = False


@app.after_request
def add_nocache_headers(response: Response) -> Response:
    """Add headers to prevent browser caching."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "-1"
    return response


def _ensure_directories() -> None:
    """Create required directories if they don't exist."""
    for directory in ["violations", "templates", "static"]:
        Path(directory).mkdir(exist_ok=True)


def _base64_to_image(base64_string: str) -> np.ndarray:
    """Convert base64 string to OpenCV image.

    Args:
        base64_string: Base64-encoded image data

    Returns:
        BGR image as numpy array
    """
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def _save_violation_screenshot(
    student_id: int, frame: np.ndarray, violation_type: str
) -> str:
    """Save screenshot with timestamp overlay.

    Args:
        student_id: Student ID
        frame: BGR image frame
        violation_type: Type of violation

    Returns:
        Screenshot filename
    """
    timestamp = int(time.time())
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add timestamp overlay
    save_frame = frame.copy()
    text = f"Time: {timestamp_str}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    height, width = save_frame.shape[:2]
    x = width - text_width - 20
    y = height - 20

    # Draw background rectangle
    cv2.rectangle(
        save_frame,
        (x - 10, y - text_height - 10),
        (x + text_width + 10, y + baseline + 10),
        (0, 0, 0),
        -1,
    )

    # Draw text
    cv2.putText(save_frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)

    # Save file
    filename = f"{violation_type}_{student_id}_{timestamp}.jpg"
    filepath = Path("violations") / filename
    cv2.imwrite(str(filepath), save_frame)

    return filename


# ============= ROUTES =============


@app.route("/")
def index() -> str:
    """Render teacher dashboard."""
    return render_template("dashboard.html")


@app.route("/student")
def student_page() -> str:
    """Render student exam interface."""
    return render_template("student.html")


# ============= API ENDPOINTS =============


@app.route("/api/student/join", methods=["POST"])
def student_join() -> tuple[dict[str, Any], int] | dict[str, Any]:
    """Register a student for the exam.

    Returns:
        JSON with student_id and name, or error
    """
    try:
        data = request.get_json()
        name = data.get("name")

        if not name:
            return {"error": "Name is required"}, 400

        student_id = db.add_student(name)
        return {"student_id": student_id, "name": name}

    except Exception as e:
        return {"error": str(e)}, 500


@app.route("/api/student/calibrate", methods=["POST"])
def calibrate() -> tuple[dict[str, Any], int] | dict[str, Any]:
    """Calibrate student's head pose baseline.

    Returns:
        JSON with calibration status and data, or error
    """
    try:
        data = request.get_json()
        student_id = data.get("student_id")
        frames_base64 = data.get("frames", [])

        if not student_id:
            return {"error": "student_id is required"}, 400

        if not frames_base64:
            return {"error": "frames are required"}, 400

        frames = [_base64_to_image(f) for f in frames_base64]
        calibration = detector.calibrate_student(student_id, frames)

        return {"status": "calibrated", "calibration": calibration}

    except ValueError as e:
        return {"error": str(e)}, 400
    except Exception as e:
        return {"error": str(e)}, 500


@app.route("/api/student/frame", methods=["POST"])
def process_frame() -> tuple[dict[str, Any], int] | dict[str, Any]:
    """Process a frame and detect violations.

    Returns:
        JSON with violations list, or error
    """
    try:
        data = request.get_json()
        student_id = data.get("student_id")
        frame_base64 = data.get("frame")

        if not student_id:
            return {"error": "student_id is required"}, 400

        if not frame_base64:
            return {"error": "frame is required"}, 400

        frame = _base64_to_image(frame_base64)
        result = detector.process_frame(student_id, frame)

        # Save violations to database
        for violation in result["violations"]:
            screenshot_path = _save_violation_screenshot(
                student_id, frame, violation["type"]
            )
            db.add_violation(
                student_id,
                violation["type"],
                screenshot_path,
                result["details"],
                confidence=violation.get("confidence", 1.0)
            )

        db.update_student_heartbeat(student_id)

        return {"violations": result["violations"]}

    except ValueError as e:
        return {"error": str(e)}, 400
    except Exception as e:
        print(f"Error processing frame: {e}")
        return {"error": str(e)}, 500


@app.route("/api/dashboard/students", methods=["GET"])
def get_students() -> tuple[dict[str, Any], int] | list[dict[str, Any]]:
    """Get all active students with violation data.

    Returns:
        JSON list of student dicts, or error
    """
    try:
        students = db.get_active_students()

        for student in students:
            student["violation_count"] = db.get_violation_count(student["id"])

            # Get highest priority violation info
            priority_violation = db.get_highest_priority_violation(student["id"])
            if priority_violation:
                student["latest_screenshot"] = priority_violation["screenshot_path"]
                student["screenshot_violation_type"] = priority_violation["violation_type"]
                student["screenshot_confidence"] = priority_violation.get("confidence", None)
            else:
                student["latest_screenshot"] = None
                student["screenshot_violation_type"] = None
                student["screenshot_confidence"] = None

        return students

    except Exception as e:
        return {"error": str(e)}, 500


@app.route("/api/dashboard/student/<int:student_id>/violations", methods=["GET"])
def get_student_violations(student_id: int) -> tuple[dict[str, Any], int] | dict[str, Any]:
    """Get violation counts by type for a specific student.

    Args:
        student_id: Student ID from URL

    Returns:
        JSON dict with violation counts by type, or error
    """
    try:
        # Get student info
        student = db.get_student_by_id(student_id)
        if not student:
            return {"error": "Student not found"}, 404

        # Get violation counts by type
        violations_by_type = db.get_violations_by_type(student_id)

        return {
            "student_id": student_id,
            "student_name": student["name"],
            "violations_by_type": violations_by_type,
            "total_violations": sum(violations_by_type.values())
        }

    except Exception as e:
        return {"error": str(e)}, 500


@app.route("/violations/<path:filename>")
def serve_screenshot(filename: str) -> Any:
    """Serve violation screenshot files.

    Args:
        filename: Screenshot filename

    Returns:
        Image file response
    """
    return send_from_directory("violations", filename)


# ============= EXAM CONTROL =============


@app.route("/api/exam/start", methods=["POST"])
def start_exam() -> dict[str, str]:
    """Start the exam - students can now join."""
    global exam_started
    exam_started = True
    return {"status": "started", "message": "Exam started!"}


@app.route("/api/exam/stop", methods=["POST"])
def stop_exam() -> dict[str, str]:
    """Stop the exam - students can no longer join."""
    global exam_started
    exam_started = False
    return {"status": "stopped", "message": "Exam stopped!"}


@app.route("/api/exam/status", methods=["GET"])
def exam_status() -> dict[str, bool]:
    """Get current exam status.

    Returns:
        JSON with 'started' boolean
    """
    return {"started": exam_started}


# ============= MAIN =============


def main() -> None:
    """Run the Flask development server."""
    _ensure_directories()

    print("=" * 50)
    print("AI PROCTORING SYSTEM - MULTI-USER DEMO")
    print("=" * 50)
    print("\nStarting server...")
    print("\nTeacher Dashboard: http://localhost:5000")
    print("Student Page: http://localhost:5000/student")
    print("\nFor other devices on same WiFi:")
    print("  1. Find your IP: ipconfig (Windows) or ifconfig (Mac/Linux)")
    print("  2. Students go to: http://YOUR-IP:5000/student")
    print(
        "  3. In Chrome, enable: chrome://flags/#unsafely-treat-insecure-origin-as-secure"
    )
    print("     Add: http://YOUR-IP:5000")
    print("\nPress Ctrl+C to stop")
    print("=" * 50)

    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)


if __name__ == "__main__":
    main()
