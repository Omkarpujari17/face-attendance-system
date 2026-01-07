from flask import Flask, request, jsonify
import cv2
import numpy as np
import pickle
import os
import json
from flask import render_template

from flask import  redirect, url_for, session

import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
# -----------------------------
# Firebase Initialization
# -----------------------------
cred_json = json.loads(os.environ["FIREBASE_CREDENTIALS"])
cred = credentials.Certificate(cred_json)

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://faceattendance-a448f-default-rtdb.firebaseio.com/'
})


# -----------------------------
# Attendance Session State
# -----------------------------
attendance_open = False

# -----------------------------
# Flask App Initialization
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Load Face Recognition Model
# -----------------------------
MODEL_PATH = os.path.join('model', 'lbph_face_model.yml')
LABEL_MAP_PATH = os.path.join('model', 'label_map.pkl')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

with open(LABEL_MAP_PATH, 'rb') as f:
    label_map = pickle.load(f)

print(" Face recognition model loaded successfully")

# -----------------------------
# Load Face Detector
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

CONFIDENCE_THRESHOLD = 70

# -----------------------------
# Home Route (Health Check)
# -----------------------------
@app.route('/')
def home():
    return "Face Attendance Backend Running"


@app.route('/student')
def student_page():
    return render_template('student.html')


# -----------------------------
# Teacher Login
# -----------------------------
app.secret_key = "secret123"  # for session

TEACHER_USERNAME = "teacher"
TEACHER_PASSWORD = "teacher123"


@app.route('/teacher', methods=['GET', 'POST'])
def teacher_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == TEACHER_USERNAME and password == TEACHER_PASSWORD:
            session['teacher'] = True
            return redirect(url_for('teacher_dashboard'))
        else:
            return render_template(
                'teacher_login.html',
                error="Invalid credentials"
            )

    return render_template('teacher_login.html')


@app.route('/dashboard')
def teacher_dashboard():
    if not session.get('teacher'):
        return redirect(url_for('teacher_login'))

    status = "OPEN" if attendance_open else "CLOSED"
    return render_template(
        'teacher_dashboard.html',
        status=status
    )


@app.route('/start_attendance', methods=['POST'])
def start_attendance():
    global attendance_open
    if session.get('teacher'):
        attendance_open = True
    return redirect(url_for('teacher_dashboard'))


@app.route('/stop_attendance', methods=['POST'])
def stop_attendance():
    global attendance_open
    if session.get('teacher'):
        attendance_open = False
    return redirect(url_for('teacher_dashboard'))


@app.route('/logout')
def logout():
    session.pop('teacher', None)
    return redirect(url_for('teacher_login'))



# -----------------------------
# Face Recognition API
# -----------------------------
@app.route('/recognize', methods=['POST'])
def recognize_face():
    
    if not attendance_open:
        return jsonify({
            'status': 'fail',
            'message': 'Attendance is closed'
        })
    # 1. Check if image is sent
    if 'image' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No image provided'
        }), 400

    file = request.files['image']
    img_bytes = file.read()

    # 2. Convert image bytes to OpenCV image
    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({
            'status': 'error',
            'message': 'Invalid image'
        }), 400

    # 3. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4. Detect face(s)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(60, 60)
    )

    # 5. Validation rules
    if len(faces) == 0:
        return jsonify({
            'status': 'fail',
            'message': 'No face detected'
        })

    if len(faces) > 1:
        return jsonify({
            'status': 'fail',
            'message': 'Multiple faces detected'
        })

    # 6. Crop and resize face
    (x, y, w, h) = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (200, 200))

    # 7. Predict using model
    label, confidence = recognizer.predict(face)

    # 8. Apply confidence threshold
    if confidence < CONFIDENCE_THRESHOLD:
        name = label_map[label]
        today = datetime.now().strftime("%Y-%m-%d")
        time_now = datetime.now().strftime("%H:%M:%S")
        ref = db.reference(f"attendance/{today}/{name}")

        if ref.get() is None:
            ref.set({
            'time': time_now,
            'confidence': float(confidence)
        })
        
        return jsonify({
        'status': 'success',
        'name': name,
        'confidence': float(confidence)
    })

    
    else:
        return jsonify({
            'status': 'fail',
            'message': 'Unknown person',
            'confidence': float(confidence)
        })

# -----------------------------
# Run Flask App
# -----------------------------
if __name__ == '__main__':
     app.run(host='0.0.0.0', port=10000)
