import os
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet

detector = MTCNN()
embedder = FaceNet()

dataset_path = "dataset/"
embeddings = []
names = []

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = detector.detect_faces(img)
        if len(faces) == 0:
            continue

        x, y, w, h = faces[0]['box']
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))

        embedding = embedder.embeddings([face])[0]
        embeddings.append(embedding)
        names.append(person)

np.savez("embeddings/face_embeddings.npz",
         embeddings=embeddings,
         names=names)

print("✅ Face embeddings generated successfully")
