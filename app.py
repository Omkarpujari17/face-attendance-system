from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import cv2
import numpy as np
import pickle
import os
import json
from datetime import datetime, date
today = date.today().strftime("%Y-%m-%d")
# -----------------------------
# Flask App Initialization
# -----------------------------
app = Flask(__name__)
app.secret_key = "secret123"   # session key

# -----------------------------
# Paths
# -----------------------------
DATASET_PATH = "dataset"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "lbph_face_model.yml")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.pkl")

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Face Recognition Config
# -----------------------------
CONFIDENCE_THRESHOLD = 65
FACE_SIZE = (200, 200)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()
label_map_global = {}

# Load model if exists
if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_MAP_PATH):
    recognizer.read(MODEL_PATH)
    with open(LABEL_MAP_PATH, "rb") as f:
        label_map_global = pickle.load(f)
    print("Face recognition model loaded successfully")
else:
    print("No trained model found. Train model first.")

# -----------------------------
# Firebase Initialization
# -----------------------------
import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate("FIREBASE_CREDENTIALS.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://faceattendance-a448f-default-rtdb.firebaseio.com/"
})

# -----------------------------
# Attendance State
# -----------------------------
attendance_open = False

# -----------------------------
# Utility: Student List
# -----------------------------
def get_student_list():
    students = []
    if not os.path.exists(DATASET_PATH):
        return students

    for student in os.listdir(DATASET_PATH):
        student_path = os.path.join(DATASET_PATH, student)
        if os.path.isdir(student_path):
            count = len([
                f for f in os.listdir(student_path)
                if f.lower().endswith((".jpg", ".png"))
            ])
            students.append({"name": student, "count": count})
    return students

def get_today_attendance():
    from datetime import date
    today = date.today().strftime("%Y-%m-%d")

    ref = db.reference(f"attendance/{today}")
    data = ref.get()

    if not data:
        return []

    return list(data.keys())

def calculate_attendance_percentage():
    ref = db.reference("attendance")
    data = ref.get() or {}

    total_days = len(data)
    if total_days == 0:
        return {}

    student_count = {}

    for day in data.values():
        for student in day.keys():
            student_count[student] = student_count.get(student, 0) + 1

    return {
        student: round((count / total_days) * 100, 2)
        for student, count in student_count.items()
    }


# -----------------------------
# Home
# -----------------------------
@app.route("/")
def home():
    return render_template("home.html")


# -----------------------------
# Teacher Login
# -----------------------------
TEACHER_USERNAME = "teacher"
TEACHER_PASSWORD = "teacher123"

@app.route("/teacher", methods=["GET", "POST"])
def teacher_login():
    if request.method == "POST":
        if (
            request.form["username"] == TEACHER_USERNAME and
            request.form["password"] == TEACHER_PASSWORD
        ):
            session["teacher"] = True
            return redirect(url_for("teacher_dashboard"))
        return render_template("teacher_login.html", error="Invalid credentials")

    return render_template("teacher_login.html")

@app.route("/logout")
def logout():
    session.pop("teacher", None)
    return redirect(url_for("teacher_login"))

# -----------------------------
# Teacher Dashboard
# -----------------------------
@app.route("/dashboard")
def teacher_dashboard():
    if not session.get("teacher"):
        return redirect(url_for("teacher_login"))

    status = "OPEN" if attendance_open else "CLOSED"
    students = get_student_list()

    today_attendance = get_today_attendance()
    percentages = calculate_attendance_percentage()

    return render_template(
        "teacher_dashboard.html",
        status=status,
        students=students,
        today_attendance=today_attendance,
        percentages=percentages
    )

@app.route("/start_attendance", methods=["POST"])
def start_attendance():
    global attendance_open
    if session.get("teacher"):
        attendance_open = True
    return redirect(url_for("teacher_dashboard"))

@app.route("/stop_attendance", methods=["POST"])
def stop_attendance():
    global attendance_open
    if session.get("teacher"):
        attendance_open = False
    return redirect(url_for("teacher_dashboard"))

# -----------------------------
# Add Student
# -----------------------------
@app.route("/add_student")
def add_student():
    if not session.get("teacher"):
        return redirect(url_for("teacher_login"))
    return render_template("add_student.html")

@app.route("/save_student_image", methods=["POST"])
def save_student_image():
    name = request.form["name"].strip().lower()
    image = request.files["image"]

    student_dir = os.path.join(DATASET_PATH, name)
    os.makedirs(student_dir, exist_ok=True)

    img_array = np.frombuffer(image.read(), np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) != 1:
        return "Image rejected (ensure single face)"

    (x, y, w, h) = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, FACE_SIZE)

    count = len(os.listdir(student_dir)) + 1
    cv2.imwrite(os.path.join(student_dir, f"{count}.jpg"), face)

    return "Saved"

# -----------------------------
# Train Model (COLAB PIPELINE)
# -----------------------------
@app.route("/train_model", methods=["POST"])
def train_model():
    global label_map_global

    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for student in os.listdir(DATASET_PATH):
        student_path = os.path.join(DATASET_PATH, student)
        if not os.path.isdir(student_path):
            continue

        print(f"Training student: {student}")
        label_map[current_label] = student

        for img_name in os.listdir(student_path):
            img_path = os.path.join(student_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in detected:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, FACE_SIZE)
                faces.append(face)
                labels.append(current_label)

        current_label += 1

    if len(faces) == 0:
        return "No faces found. Training aborted."

    recognizer.train(np.array(faces), np.array(labels))
    recognizer.save(MODEL_PATH)

    with open(LABEL_MAP_PATH, "wb") as f:
        pickle.dump(label_map, f)

    # Reload into memory
    recognizer.read(MODEL_PATH)
    with open(LABEL_MAP_PATH, "rb") as f:
        label_map_global = pickle.load(f)

    print("Training completed successfully")
    return redirect(url_for("teacher_dashboard"))

@app.route("/attendance_sheet")
def attendance_sheet():
    if not session.get("teacher"):
        return redirect(url_for("teacher_login"))

    ref = db.reference("attendance")
    data = ref.get() or {}

    return render_template("attendance_sheet.html", attendance=data)

# -----------------------------
# Student Page
# -----------------------------
@app.route("/student")
def student_page():
    return render_template("student.html")

# -----------------------------
# Face Recognition
# -----------------------------
@app.route("/recognize", methods=["POST"])
def recognize():
    if not attendance_open:
        return jsonify({"status": "fail", "message": "Attendance is closed"})

    image = request.files["image"]
    img_array = np.frombuffer(image.read(), np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) != 1:
        return jsonify({"status": "fail", "message": "Ensure exactly one face is visible"})

    (x, y, w, h) = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, FACE_SIZE)

    label, confidence = recognizer.predict(face)
    print("Prediction:", label, confidence)

    if confidence > CONFIDENCE_THRESHOLD:
        return jsonify({
            "status": "fail",
            "message": "Unknown person",
            "confidence": float(confidence)
        })

    name = label_map_global.get(label, "Unknown")

    # Save attendance
    ref = db.reference(f"attendance/{today}")
    ref.child(name).set(True)
    print("DEBUG: Saving attendance")
    print("DEBUG Date:", today)
    print("DEBUG Student:", name)

    return jsonify({
        "status": "success",
        "name": name,
        "confidence": float(confidence)
    })

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
