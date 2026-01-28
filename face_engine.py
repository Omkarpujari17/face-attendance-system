import torch
import numpy as np
import cv2
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# -------------------------------
# Device
# -------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------------------
# Face Detector (MTCNN)
# -------------------------------
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    min_face_size=40,
    device=device
)

# -------------------------------
# Face Recognition Model (FaceNet)
# -------------------------------
facenet = InceptionResnetV1(
    pretrained='vggface2'
).eval().to(device)

# -------------------------------
# Load / Create Embedding DB
# -------------------------------
EMBEDDING_PATH = 'embeddings/students.pkl'

if not os.path.exists('embeddings'):
    os.makedirs('embeddings')

if os.path.exists(EMBEDDING_PATH):
    with open(EMBEDDING_PATH, 'rb') as f:
        database = pickle.load(f)
else:
    database = {}

# -------------------------------
# Get Face Embedding
# -------------------------------
def get_embedding(frame):
    face = mtcnn(frame)
    if face is None:
        return None

    face = face.unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = facenet(face)

    return embedding.cpu().numpy()[0]

# -------------------------------
# Register Face
# -------------------------------
def register_face(student_id, frame):
    embedding = get_embedding(frame)
    if embedding is None:
        return False

    # If student not exists, create list
    if student_id not in database:
        database[student_id] = []

    # Add new embedding (each photo = one angle)
    database[student_id].append(embedding)

    with open(EMBEDDING_PATH, 'wb') as f:
        pickle.dump(database, f)

    return True


# -------------------------------
# Recognize Face
# -------------------------------
def recognize_face(embedding, threshold=0.55):
    if embedding is None or len(database) == 0:
        return "Unknown", None

    best_match = "Unknown"
    best_score = -1

    for student_id, embeddings_list in database.items():
        for db_embedding in embeddings_list:
            score = cosine_similarity(
                [embedding], [db_embedding]
            )[0][0]

            if score > best_score:
                best_score = score
                best_match = student_id

    if best_score >= threshold:
        return best_match, best_score
    else:
        return "Unknown", best_score
