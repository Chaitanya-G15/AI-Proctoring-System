// Student exam page JavaScript
// Handles webcam capture, calibration, and frame processing

let studentId = null;
let video = null;
let detectionInterval = null;
let statusCheckInterval = null;

// ============= CHECK EXAM STATUS =============

async function checkExamStatus() {
    try {
        const response = await fetch('/api/exam/status');
        const data = await response.json();

        if (data.started) {
            // Exam started - show join screen
            document.getElementById('waitingScreen').style.display = 'none';
            document.getElementById('joinScreen').style.display = 'block';

            // Stop checking
            if (statusCheckInterval) {
                clearInterval(statusCheckInterval);
            }
        } else {
            // Exam not started - show waiting screen
            document.getElementById('joinScreen').style.display = 'none';
            document.getElementById('waitingScreen').style.display = 'block';
        }
    } catch (error) {
        console.error('Error checking exam status:', error);
    }
}

// Check status on page load and every 3 seconds
window.addEventListener('DOMContentLoaded', () => {
    checkExamStatus();
    statusCheckInterval = setInterval(checkExamStatus, 3000);
});

// ============= JOIN EXAM =============

async function joinExam() {
    const name = document.getElementById('studentName').value.trim();

    if (!name) {
        alert('Please enter your name');
        return;
    }

    try {
        // Register student
        const response = await fetch('/api/student/join', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });

        if (!response.ok) {
            const error = await response.json();
            alert('Error joining exam: ' + error.error);
            return;
        }

        const data = await response.json();
        studentId = data.student_id;

        console.log('Joined exam as student ID:', studentId);

        // Hide join screen, show calibration
        document.getElementById('joinScreen').style.display = 'none';
        document.getElementById('calibrationScreen').style.display = 'block';

        // Start webcam
        await startWebcam();

        // Start calibration
        await calibrate();

    } catch (error) {
        console.error('Error joining exam:', error);
        alert('Failed to join exam. Please try again.');
    }
}

// ============= WEBCAM =============

async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' }
        });

        video = document.getElementById('video');
        const calibrationVideo = document.getElementById('calibrationVideo');

        calibrationVideo.srcObject = stream;
        video.srcObject = stream;

        console.log('Webcam started');

    } catch (error) {
        console.error('Error accessing webcam:', error);
        alert('Unable to access webcam. Please grant camera permissions and try again.');
        throw error;
    }
}

// ============= CALIBRATION =============

async function calibrate() {
    const frames = [];
    const calibrationVideo = document.getElementById('calibrationVideo');
    const timerEl = document.getElementById('calibrationTimer');

    console.log('Starting calibration...');

    // Wait for video to be ready
    await new Promise(resolve => {
        if (calibrationVideo.readyState === calibrationVideo.HAVE_ENOUGH_DATA) {
            resolve();
        } else {
            calibrationVideo.addEventListener('loadeddata', resolve, { once: true });
        }
    });

    // Capture 30 frames over 3 seconds
    for (let i = 3; i > 0; i--) {
        timerEl.textContent = i;
        await sleep(1000);

        // Capture 10 frames per second
        for (let j = 0; j < 10; j++) {
            const frame = captureFrame(calibrationVideo);
            if (frame) {
                frames.push(frame);
            }
            await sleep(100);
        }
    }

    console.log(`Captured ${frames.length} calibration frames`);

    // Send calibration data
    try {
        const response = await fetch('/api/student/calibrate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                student_id: studentId,
                frames: frames
            })
        });

        if (!response.ok) {
            const error = await response.json();
            alert('Calibration failed: ' + error.error);
            return;
        }

        const data = await response.json();
        console.log('Calibration successful:', data.calibration);

        // Hide calibration, show exam
        document.getElementById('calibrationScreen').style.display = 'none';
        document.getElementById('examScreen').style.display = 'block';

        // Start detection loop
        startDetection();

    } catch (error) {
        console.error('Calibration error:', error);
        alert('Calibration failed. Please reload and try again.');
    }
}

// ============= FRAME CAPTURE =============

function captureFrame(videoElement) {
    try {
        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;

        if (canvas.width === 0 || canvas.height === 0) {
            console.warn('Video not ready yet');
            return null;
        }

        const ctx = canvas.getContext('2d');

        // Mirror the frame horizontally to match what student sees
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(videoElement, 0, 0);

        // Convert to base64 (JPEG, 80% quality) and remove the prefix
        const dataURL = canvas.toDataURL('image/jpeg', 0.8);
        const base64 = dataURL.split(',')[1];

        return base64;

    } catch (error) {
        console.error('Error capturing frame:', error);
        return null;
    }
}

// ============= DETECTION LOOP =============

function startDetection() {
    console.log('Starting detection loop (1 frame/second)...');

    // Send frame every 1 second
    detectionInterval = setInterval(async () => {
        const frame = captureFrame(video);

        if (!frame) {
            console.warn('Frame capture failed');
            return;
        }

        try {
            const response = await fetch('/api/student/frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    student_id: studentId,
                    frame: frame
                })
            });

            if (!response.ok) {
                console.error('Frame processing error:', response.status);
                return;
            }

            const data = await response.json();

            // Show violations (use all_violations to show warnings immediately)
            const violationsToShow = data.all_violations || data.violations || [];
            if (violationsToShow.length > 0) {
                showViolation(violationsToShow[0].type);
            } else {
                hideViolation();
            }

        } catch (error) {
            console.error('Error sending frame:', error);
        }

    }, 1000);  // 1 second interval
}

// ============= VIOLATION ALERTS =============

function showViolation(type) {
    const badge = document.getElementById('statusBadge');
    const alert = document.getElementById('violationAlert');

    badge.className = 'badge-warning';
    badge.textContent = 'Warning!';

    const messages = {
        'not_focused': '⚠️ Please look at the screen!',
        'phone': '⚠️ Phone detected! Put it away immediately.',
        'multiple_people': '⚠️ Multiple people detected! Only you should be visible.'
    };

    alert.textContent = messages[type] || '⚠️ Violation detected';
    alert.style.display = 'block';

    // Play beep sound
    playBeep();
}

function hideViolation() {
    const badge = document.getElementById('statusBadge');
    const alert = document.getElementById('violationAlert');

    badge.className = 'badge-focused';
    badge.textContent = 'Focused';
    alert.style.display = 'none';
}

function playBeep() {
    // Simple beep sound (optional - might not work in all browsers)
    try {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);

        oscillator.frequency.value = 800;  // 800 Hz
        oscillator.type = 'sine';

        gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);

        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.3);
    } catch (error) {
        // Audio might be blocked by browser
        console.log('Audio playback blocked');
    }
}

// ============= EXAM SUBMISSION =============

function submitExam() {
    if (confirm('Are you sure you want to submit your exam?')) {
        // Stop detection
        if (detectionInterval) {
            clearInterval(detectionInterval);
        }

        // Stop webcam
        if (video && video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
        }

        alert('Exam submitted! You may now close this window.');
        document.getElementById('examScreen').innerHTML = '<div class="submit-message"><h2>Exam Submitted Successfully</h2><p>You may now close this window.</p></div>';
    }
}

// ============= UTILITIES =============

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (detectionInterval) {
        clearInterval(detectionInterval);
    }
    if (video && video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }
});
