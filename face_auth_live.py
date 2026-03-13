import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import time
from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine
from email_alert import send_sms_alert

# -----------------------
# Initialize Models
# -----------------------
detector = MTCNN()
embedder = FaceNet()

# -----------------------
# Load Stored Embeddings
# -----------------------
data = np.load("embeddings/face_embeddings.npz", allow_pickle=True)

known_embeddings = data["embeddings"]
known_names = data["names"]

assert len(known_embeddings) > 0, "❌ No face embeddings found!"

# -----------------------
# Settings
# -----------------------
THRESHOLD = 0.4
ALERT_COOLDOWN = 30
MIN_FACE_SIZE = 80
CONFIDENCE_THRESHOLD = 0.90
last_alert_time = 0

os.makedirs("unknown_faces", exist_ok=True)

# -----------------------
# Camera Setup
# -----------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

print("✅ Camera started")

# -----------------------
# Main Loop
# -----------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    try:
        faces = detector.detect_faces(rgb)
    except Exception as e:
        print(f"⚠️ Detection error: {e}")
        cv2.imshow("Face Authentication System", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    # Prevent crash when no faces detected
    if faces is None or len(faces) == 0:
        cv2.imshow("Face Authentication System", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    for face in faces:

        # Skip low-confidence detections
        if face.get('confidence', 1.0) < CONFIDENCE_THRESHOLD:
            continue

        x, y, w, h = face['box']

        # Clamp all values to non-negative
        x = max(0, x)
        y = max(0, y)
        w = max(0, w)
        h = max(0, h)

        x2 = min(x + w, rgb.shape[1])
        y2 = min(y + h, rgb.shape[0])

        # Skip faces that are too small after clipping
        if (x2 - x) < MIN_FACE_SIZE or (y2 - y) < MIN_FACE_SIZE:
            continue

        face_img = rgb[y:y2, x:x2]

        # Skip empty crops
        if face_img.size == 0:
            continue

        face_img = cv2.resize(face_img, (160, 160))

        # Generate embedding
        try:
            emb = embedder.embeddings([face_img])[0]
        except Exception as e:
            print(f"⚠️ Embedding error: {e}")
            continue

        # Compare embeddings
        distances = np.array([cosine(emb, e) for e in known_embeddings])

        min_dist = distances.min()

        if min_dist < THRESHOLD:

            name = known_names[distances.argmin()]
            label = f"Authenticated: {name}"
            color = (0, 255, 0)

        else:

            label = "Unauthenticated"
            color = (0, 0, 255)

            # Save unknown face
            timestamp = int(time.time() * 1000)
            img_path = f"unknown_faces/unknown_{timestamp}.jpg"
            cv2.imwrite(img_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))

            # Send email alert with cooldown
            if time.time() - last_alert_time > ALERT_COOLDOWN:

                print("⚠️ Unknown face detected! Sending email alert...")

                try:
                    send_sms_alert()
                    print("📧 Email alert sent")
                except Exception as e:
                    print(f"⚠️ Alert failed: {e}")

                last_alert_time = time.time()

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

        # Draw label background for readability
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(
            frame,
            (x, y - label_size[1] - 15),
            (x + label_size[0] + 5, y),
            color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Show confidence score
        conf_text = f"Conf: {face.get('confidence', 0):.2f} | Dist: {min_dist:.2f}"
        cv2.putText(
            frame,
            conf_text,
            (x, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    cv2.imshow("Face Authentication System", frame)

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break


# -----------------------
# Cleanup
# -----------------------
cap.release()
cv2.destroyAllWindows()