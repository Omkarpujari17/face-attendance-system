import cv2
from face_engine import register_face

student_id = "omkar_001"   # change for each student

cap = cv2.VideoCapture(0)
print("Press 's' to save face | 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Register Face", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        success = register_face(student_id, frame)
        print("Face saved successfully!" if success else "Face not detected")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
