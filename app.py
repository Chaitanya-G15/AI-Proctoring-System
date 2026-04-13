from flask import Flask, request, render_template, jsonify, send_from_directory
import cv2
import numpy as np
import base64
import os
import time
from datetime import datetime
from detection import ProctorDetector
from database import Database

app = Flask(__name__)

# Initialize detector and database
detector = ProctorDetector()
db = Database()

# Create necessary folders
os.makedirs('violations', exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)

# ============= HELPER FUNCTIONS =============

def base64_to_cv2(base64_string):
    """Convert base64 string to CV2 image."""
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def save_screenshot(student_id, frame, violation_type):
    """Save screenshot to violations folder."""
    timestamp = int(time.time())
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add timestamp overlay
    save_frame = frame.copy()
    text = f"Time: {timestamp_str}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    # Get text size for background
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Position at bottom-right
    height, width = save_frame.shape[:2]
    x = width - text_width - 20
    y = height - 20

    # Draw background rectangle
    cv2.rectangle(save_frame, (x - 10, y - text_height - 10),
                  (x + text_width + 10, y + baseline + 10), (0, 0, 0), -1)

    # Draw text
    cv2.putText(save_frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)

    # Save
    filename = f"{violation_type}_{student_id}_{timestamp}.jpg"
    filepath = os.path.join('violations', filename)
    cv2.imwrite(filepath, save_frame)

    return filename

# ============= ROUTES =============

@app.route('/')
def index():
    """Redirect to dashboard."""
    return render_template('dashboard.html')

@app.route('/student')
def student_page():
    """Student exam interface."""
    return render_template('student.html')

# ============= API ENDPOINTS =============

@app.route('/api/student/join', methods=['POST'])
def student_join():
    """
    Register a student for the exam session.

    Request JSON:
        {
            "name": "Student Name"
        }

    Returns:
        {
            "student_id": 1,
            "name": "Student Name"
        }
    """
    try:
        data = request.get_json()
        name = data.get('name')

        if not name:
            return jsonify({'error': 'Name is required'}), 400

        student_id = db.add_student(name)

        return jsonify({
            'student_id': student_id,
            'name': name
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/student/calibrate', methods=['POST'])
def calibrate():
    """
    Calibrate a student's head pose.

    Request JSON:
        {
            "student_id": 1,
            "frames": ["base64_frame1", "base64_frame2", ...]
        }

    Returns:
        {
            "status": "calibrated",
            "calibration": {
                "pitch": 100.5,
                "yaw": 50.2,
                "samples": 30
            }
        }
    """
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        frames_base64 = data.get('frames', [])

        if not student_id:
            return jsonify({'error': 'student_id is required'}), 400

        if not frames_base64:
            return jsonify({'error': 'frames are required'}), 400

        # Decode frames
        frames = [base64_to_cv2(f) for f in frames_base64]

        # Calibrate
        calibration = detector.calibrate_student(student_id, frames)

        return jsonify({
            'status': 'calibrated',
            'calibration': calibration
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/student/frame', methods=['POST'])
def process_frame():
    """
    Process a frame and detect violations.

    Request JSON:
        {
            "student_id": 1,
            "frame": "base64_frame_data"
        }

    Returns:
        {
            "violations": [
                {
                    "type": "not_focused",
                    "confidence": 1.0
                }
            ]
        }
    """
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        frame_base64 = data.get('frame')

        if not student_id:
            return jsonify({'error': 'student_id is required'}), 400

        if not frame_base64:
            return jsonify({'error': 'frame is required'}), 400

        # Decode frame
        frame = base64_to_cv2(frame_base64)

        # Run detection
        result = detector.process_frame(student_id, frame)

        # Save violations to database
        for violation in result['violations']:
            screenshot_path = save_screenshot(student_id, frame, violation['type'])
            db.add_violation(
                student_id,
                violation['type'],
                screenshot_path,
                result['details']
            )

        # Update last_seen timestamp
        db.update_student_heartbeat(student_id)

        return jsonify({
            'violations': result['violations']
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard/students', methods=['GET'])
def get_students():
    """
    Get all active students with their violation counts.

    Returns:
        [
            {
                "id": 1,
                "name": "Student Name",
                "joined_at": "2025-01-15 10:30:00",
                "last_seen": "2025-01-15 10:35:00",
                "violation_count": 5,
                "latest_screenshot": "not_focused_1_1234567890.jpg"
            }
        ]
    """
    try:
        students = db.get_active_students()

        # Enrich with violation data
        for student in students:
            student['violation_count'] = db.get_violation_count(student['id'])
            student['latest_screenshot'] = db.get_latest_screenshot(student['id'])

        return jsonify(students)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/violations/<path:filename>')
def serve_screenshot(filename):
    """Serve violation screenshots."""
    return send_from_directory('violations', filename)

# ============= MAIN =============

if __name__ == '__main__':
    print("=" * 50)
    print("AI PROCTORING SYSTEM - MULTI-USER DEMO")
    print("=" * 50)

    # print("\n⚠️  HTTP MODE (Webcam only works on localhost)")
    # print("\nTo enable HTTPS for multi-device demo:")
    # print("  1. Install: pip install pyopenssl")
    # print("  2. Run: python generate_cert.py")
    # print("  3. Restart: python app.py")
    print(f"\nTeacher Dashboard: http://localhost:5000")
    print(f"Student Page: http://localhost:5000/student")



    app.run(host='0.0.0.0', port=5000, debug=True)
