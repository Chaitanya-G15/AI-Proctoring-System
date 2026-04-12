# CV2 + YOLO Hybrid - AI Proctoring System

A hybrid version that uses **OpenCV's Haar Cascades** for face/eye detection and **YOLO** for phone/book/earphone detection.

## Features

### ✅ What's Included
- 👤 **Face Detection** - OpenCV Haar Cascade frontal face detection
- 👁️ **Eye Detection** - Detects if eyes are visible (looking at camera)
- 👥 **Multiple Person Detection** - Alerts when more than one face detected
- 📱 **Phone Detection** - YOLO-based cell phone detection
- 📖 **Book Detection** - YOLO-based book detection
- 🎧 **Earphone Detection** - YOLO-based earphone/headphone detection
- ⌚ **Smart Watch Detection** - YOLO-based smartwatch detection
- 🔔 **Alert System** - Visual and audio alerts for violations
- 📸 **Screenshot Capture** - Saves timestamped screenshots to `violations/` folder with 10-second cooldown
- 📊 **Session Summary** - Shows percentage statistics for all violations
- ⏱️ **Cooldown System** - Prevents data burst by limiting screenshots to one per violation type every 10 seconds

## Technology Stack

- **OpenCV Haar Cascades** - Fast, built-in face and eye detection
  - No external model downloads needed
  - Pre-trained cascades included with OpenCV
- **YOLOv8** - Object detection for phones, books, earphones, and smart watches
  - Requires yolov8n.pt model download
  - Auto-downloads on first run
  - Detects from COCO dataset classes
- **No MediaPipe** - Simpler alternative to the full version
- **Smart Cooldown System** - Prevents duplicate screenshots within 10 seconds of same violation type

## Advantages

- **No MediaPipe dependency** - Uses OpenCV's built-in classifiers
- **No separate face model downloads** - Haar Cascades included with OpenCV
- **Fast face detection** - Haar Cascades are very efficient
- **Complete object detection** - YOLO for phones, books, earphones, and smart watches
- **Smart storage management** - 10-second cooldown prevents duplicate screenshots
- **Efficient evidence collection** - Timestamps allow invigilators to verify without overwhelming data
- **Simple setup** - Fewer dependencies than the full version

## Detection Methods

### Face Detection (OpenCV)
- Uses Haar Cascade frontal face detector
- Draws green rectangles around detected faces
- Triggers alert if no face detected for >15 frames

### Eye Detection (OpenCV)
- Uses Haar Cascade eye detector
- Searches for eyes in upper half of detected face
- Requires at least 2 eyes detected to be "focused"
- Triggers alert if no eyes detected for >10 frames

### Object Detection (YOLO)
- Detects "cell phone", "book", "earphones/headphones", and "watch"
- Draws red rectangles with labels
- Real-time detection on every frame
- **COCO classes used:**
  - "cell phone" - Mobile phones ✅
  - "book" - Physical books ✅
  - "watch" - Wristwatches/smartwatches ✅
  - "earphones/headphones" - Audio devices ⚠️ (may not be in standard COCO)

### Screenshot Cooldown System
- **Purpose:** Prevents data burst and storage overflow
- **Mechanism:** Each violation type has independent 10-second cooldown
- **Benefit:** Still shows real-time alerts but only saves one screenshot per violation type every 10 seconds
- **Customizable:** Adjust `SCREENSHOT_COOLDOWN` variable in code (default: 10 seconds)
- **Invigilator-friendly:** Timestamps in screenshots allow precise verification without reviewing hundreds of duplicate images

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

3. Download YOLO model:
```bash
# The model will be auto-downloaded on first run
# Or manually download:
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Usage

Run the application:
```bash
python main.py
```

### Steps:
1. System starts monitoring immediately (no calibration needed)
2. Green rectangles show detected faces
3. Blue rectangles show detected eyes
4. Red rectangles show detected phones/books/earphones/watches
5. Console prints "📸 Screenshot saved" when cooldown allows
6. Press **'q'** to exit and see session summary

## Violations Detected

1. **No Face** - Face not visible for >15 frames
2. **Not Focused** - Eyes not detected (looking down/away) for >10 frames
3. **Multiple People** - More than one face detected
4. **Phone** - Cell phone detected by YOLO
5. **Earphones** - Earphones/headphones detected by YOLO
6. **Watch** - Smart watch detected by YOLO

**Note:** Each violation type has independent 10-second cooldown for screenshots. Alerts still appear every frame, but screenshots are saved only once per 10 seconds per violation type.

## Screenshot Storage

All violations are automatically saved to the `violations/` folder with:
- Violation type label
- Timestamp overlay (bottom-right corner)
- Unix timestamp in filename

Example: `violations/phone_1775921570.jpg`

## Thresholds & Configuration

You can adjust these in the code:

```python
NO_FACE_THRESHOLD = 15       # Frames before "no face" alert
NO_EYES_THRESHOLD = 10       # Frames before "not focused" alert
ALERT_DELAY = 2              # Seconds between audio beeps
SCREENSHOT_COOLDOWN = 10     # Seconds between saving same violation type
```

**Cooldown Examples:**
- If phone detected at 10:00:00, next phone screenshot can be saved at 10:00:10
- Different violations are independent (phone at 10:00:00, earphone at 10:00:01 both save)
- Alerts still appear every frame regardless of cooldown

## Platform Notes

- **Windows** - Full support including audio alerts
- **Linux/Mac** - Works but audio alerts are disabled (winsound not available)

## Comparison with Other Versions

| Feature | CV2+YOLO (This) | MediaPipe Only | Full Version |
|---------|-----------------|----------------|--------------|
| Face Detection | CV2 Haar Cascade | MediaPipe | MediaPipe |
| Eye Detection | CV2 Haar Cascade | MediaPipe Iris | MediaPipe Iris |
| Head Pose Tracking | ❌ | ✅ | ✅ |
| Phone Detection | ✅ YOLO | ❌ | ✅ YOLO |
| Book Detection | ✅ YOLO | ❌ | ✅ YOLO |
| Earphone Detection | ✅ YOLO | ❌ | ❌ |
| Smart Watch Detection | ✅ YOLO | ❌ | ❌ |
| Screenshot Cooldown | ✅ 10 sec | ❌ | ❌ |
| Calibration Required | ❌ No | ✅ Yes | ✅ Yes |
| Installation Size | ~6GB | ~500MB | ~6.5GB |
| Speed | Fast | Fastest | Moderate |
| Accuracy | Good | Better | Best |

## When to Use This Version

Choose this CV2 + YOLO hybrid if:
- ✅ You need phone/book/earphone/watch detection
- ✅ You want smart storage management (cooldown system)
- ✅ You want faster setup than MediaPipe
- ✅ You prefer simpler face detection
- ✅ You don't need precise head pose tracking
- ✅ You want no calibration phase

Use MediaPipe versions if:
- You need precise head pose angles
- You want iris-based eye tracking
- You don't need object detection

## Limitations

- **Less precise** than MediaPipe for head pose estimation
- **Haar Cascades** can have false positives in poor lighting
- **Front-facing only** - Won't detect faces at extreme angles
- **Eye detection** less accurate than iris tracking
- **Earphone detection** - Standard YOLOv8n (COCO) may not detect earphones well; consider using YOLOv8x or custom-trained model for better results

## Improving Earphone Detection

The standard COCO dataset (used by YOLOv8n) doesn't include "earphones" or "headphones" as a class. To improve detection:

1. **Use a larger YOLO model**: Try YOLOv8m or YOLOv8x (slower but more accurate)
   ```python
   yolo_model = YOLO('yolov8x.pt')  # Instead of yolov8n.pt
   ```

2. **Use a custom-trained model**: Train YOLO on a dataset that includes earphones/headphones

3. **Alternative detection**: Some earphones might be detected under other COCO classes like "remote" or "tie" - experiment with different class names in the code

## Requirements

- Working webcam
- Good lighting conditions (important for Haar Cascades)
- Python 3.8+
- PyTorch (auto-installed with Ultralytics)
