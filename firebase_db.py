import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate("firebase_key.json")

firebase_admin.initialize_app(cred, {
    "databaseURL": "https://face-attendance-system-33903-default-rtdb.firebaseio.com/"
})

# -------- Attendance Status --------
def get_attendance_status():
    ref = db.reference("attendance_status/status")
    return ref.get() or "OFF"

def set_attendance_status(status):
    ref = db.reference("attendance_status")
    ref.set({"status": status})

# -------- Attendance Records --------
def mark_attendance(date, student_id):
    ref = db.reference(f"attendance/{date}/{student_id}")
    ref.set("Present")

def is_already_marked(date, student_id):
    ref = db.reference(f"attendance/{date}/{student_id}")
    return ref.get() is not None

def get_all_attendance():
    ref = db.reference("attendance")
    return ref.get() or {}
