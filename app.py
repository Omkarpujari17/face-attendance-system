from flask import Flask, render_template, request, redirect, session, jsonify
import cv2
import numpy as np
import base64
from datetime import date
import json
import os
import pandas as pd
from flask import send_file
from datetime import datetime


# Face recognition engine
from face_engine import get_embedding, recognize_face, register_face

# Firebase
import firebase_admin
from firebase_admin import credentials, db

# =============================
# FLASK APP
# =============================
app = Flask(__name__)
app.secret_key = "secret123"

# =============================
# FIREBASE INIT
# =============================
# cred = credentials.Certificate("firebase_key.json")

# firebase_admin.initialize_app(cred, {
#     "databaseURL": "https://face-attendance-system-33903-default-rtdb.firebaseio.com//"
# })


# import json
# from firebase_admin import credentials

# firebase_key = json.loads(os.environ.get("FIREBASE_KEY"))
# cred = credentials.Certificate(firebase_key)


import os, json
import firebase_admin
from firebase_admin import credentials, db

firebase_key_path = os.environ.get("FIREBASE_KEY")

with open(firebase_key_path, "r") as f:
    firebase_key = json.load(f)

cred = credentials.Certificate(firebase_key)

firebase_admin.initialize_app(cred, {
    "databaseURL": "https://face-attendance-system-33903-default-rtdb.firebaseio.com/"
})

# =============================
# LOCAL FILE (TEACHERS ONLY)
# =============================
TEACHERS_FILE = "data/teachers.json"

# =============================
# UTIL FUNCTIONS
# =============================
def load_teachers():
    with open(TEACHERS_FILE, "r") as f:
        return json.load(f)

# =============================
# HOME
# =============================
@app.route("/")
def index():
    return render_template("index.html")

# =====================================================
#                    TEACHER
# =====================================================

@app.route("/teacher_login", methods=["GET", "POST"])
def teacher_login():
    if request.method == "POST":
        teacher_id = request.form["teacher_id"]
        password = request.form["password"]

        teachers = load_teachers()

        if teacher_id in teachers and teachers[teacher_id]["password"] == password:
            session["teacher"] = teacher_id
            return redirect("/teacher_dashboard")
        else:
            return render_template("teacher_login.html", error="Invalid credentials")

    return render_template("teacher_login.html")


@app.route("/teacher_dashboard")
def teacher_dashboard():
    if "teacher" not in session:
        return redirect("/teacher_login")

    status = db.reference("attendance_status/status").get() or "OFF"
    teacher_name = load_teachers()[session["teacher"]]["name"]

    return render_template(
        "teacher_dashboard.html",
        teacher_name=teacher_name,
        status=status
    )


@app.route("/start_attendance", methods=["POST"])
def start_attendance():
    if "teacher" in session:
        db.reference("attendance_status").set({"status": "ON"})
    return redirect("/teacher_dashboard")


@app.route("/stop_attendance", methods=["POST"])
def stop_attendance():
    if "teacher" in session:
        db.reference("attendance_status").set({"status": "OFF"})
    return redirect("/teacher_dashboard")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# =====================================================
#        TEACHER → REGISTER STUDENT (CAMERA)
# =====================================================

@app.route("/register_student", methods=["GET"])
def register_student_page():
    if "teacher" not in session:
        return redirect("/teacher_login")
    return render_template("register_student.html")


@app.route("/register_student", methods=["POST"])
def register_student():
    if "teacher" not in session:
        return jsonify({"message": "Unauthorized ❌"})

    data = request.json
    student_id = data["student_id"]
    student_name = data["student_name"]
    image_data = data["image"]

    # Decode image
    encoded = image_data.split(",")[1]
    img_bytes = base64.b64decode(encoded)

    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Save face embedding
    success = register_face(student_id, frame)

    if not success:
        return jsonify({"message": "Face not detected ❌"})

    # Save student details in Firebase
    db.reference(f"students/{student_id}").set({
        "name": student_name
    })

    return jsonify({
        "message": f"{student_name} ({student_id}) registered successfully ✅"
    })



@app.route("/download_attendance_today")
def download_attendance_today():
    if "teacher" not in session:
        return redirect("/teacher_login")

    today = date.today().strftime("%Y-%m-%d")
    attendance = db.reference(f"attendance/{today}").get()
    students = db.reference("students").get() or {}

    if not attendance:
        return "No attendance for today"

    data = []

    for student_id, status in attendance.items():
        student_name = students.get(student_id, {}).get("name", "Unknown")

        data.append({
            "Date": today,
            "Student ID": student_id,
            "Student Name": student_name,
            "Status": status
        })

    df = pd.DataFrame(data)

    filename = f"attendance_{today}.xlsx"
    df.to_excel(filename, index=False)

    return send_file(filename, as_attachment=True)


@app.route("/download_attendance_month")
def download_attendance_month():
    if "teacher" not in session:
        return redirect("/teacher_login")

    current_month = datetime.now().strftime("%Y-%m")
    attendance = db.reference("attendance").get()
    students = db.reference("students").get() or {}

    if not attendance:
        return "No attendance data"

    data = []

    for day, records in attendance.items():
        if day.startswith(current_month):
            for student_id, status in records.items():
                student_name = students.get(student_id, {}).get("name", "Unknown")

                data.append({
                    "Date": day,
                    "Student ID": student_id,
                    "Student Name": student_name,
                    "Status": status
                })

    if not data:
        return "No attendance for this month"

    df = pd.DataFrame(data)

    filename = f"attendance_{current_month}.xlsx"
    df.to_excel(filename, index=False)

    return send_file(filename, as_attachment=True)


@app.route("/attendance_percentage")
def attendance_percentage():
    if "teacher" not in session:
        return redirect("/teacher_login")

    # Fetch attendance
    attendance = db.reference("attendance").get() or {}
    students = db.reference("students").get() or {}

    total_days = len(attendance)

    student_count = {}

    # Count presence
    for day, records in attendance.items():
        for student_id in records.keys():
            student_count[student_id] = student_count.get(student_id, 0) + 1

    report = []

    for student_id, present_days in student_count.items():
        student_name = students.get(student_id, {}).get("name", "Unknown")

        percentage = round((present_days / total_days) * 100, 2) if total_days > 0 else 0

        report.append({
            "student_id": student_id,
            "student_name": student_name,
            "present_days": present_days,
            "total_days": total_days,
            "percentage": percentage
        })

    return render_template(
        "attendance_percentage.html",
        report=report,
        total_days=total_days
    )

# =====================================================
#                    STUDENT
# =====================================================

@app.route("/student")
def student_page():
    return render_template("student_attendance.html")


@app.route("/mark_student_attendance", methods=["POST"])
def mark_student_attendance():
    status = db.reference("attendance_status/status").get()

    if status != "ON":
        return jsonify({"message": "Attendance is CLOSED ❌"})

    data = request.json["image"]
    encoded = data.split(",")[1]
    img_bytes = base64.b64decode(encoded)

    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    embedding = get_embedding(frame)
    name, score = recognize_face(embedding)

    if name == "Unknown":
        return jsonify({"message": "Face not recognized ❌"})

    today = date.today().strftime("%Y-%m-%d")
    ref = db.reference(f"attendance/{today}/{name}")

    if ref.get():
        return jsonify({"message": f"{name} already marked ✅"})

    ref.set("Present")

    return jsonify({"message": f"Attendance marked for {name} ✅"})

# =====================================================
#              VIEW ATTENDANCE (TEACHER)
# =====================================================

@app.route("/view_attendance")
def view_attendance():
    if "teacher" not in session:
        return redirect("/teacher_login")

    attendance = db.reference("attendance").get() or {}
    return render_template("attendance_view.html", attendance=attendance)

# =============================
# RUN
# =============================
if __name__ == "__main__":
    app.run()

