import cv2
from face_engine import get_embedding, recognize_face

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    embedding = get_embedding(frame)
    name, score = recognize_face(embedding)

    text = name if score is None else f"{name} ({score:.2f})"

    cv2.putText(
        frame, text,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
