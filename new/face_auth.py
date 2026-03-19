"""
face_auth.py — Face Detection & Authentication
Uses MTCNN for detection + FaceNet for embeddings + cosine similarity
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
from scipy.spatial.distance import cosine


class FaceAuthenticator:
    def __init__(
        self,
        embeddings_path: str = "embeddings/face_embeddings.npz",
        threshold: float = 0.4,
        min_face_size: int = 80,
        confidence_threshold: float = 0.90,
    ):
        from mtcnn import MTCNN
        from keras_facenet import FaceNet

        self.detector = MTCNN()
        self.embedder = FaceNet()
        self.threshold = threshold
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold

        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                f"Embeddings not found at '{embeddings_path}'. "
                "Run your enrollment script first."
            )

        data = np.load(embeddings_path, allow_pickle=True)
        self.known_embeddings = data["embeddings"]
        self.known_names = data["names"]

        if len(self.known_embeddings) == 0:
            raise ValueError("Embedding file is empty — no known faces loaded.")

        print(f"[FaceAuth] Loaded {len(self.known_names)} known identities.")

    # ------------------------------------------------------------------
    def authenticate(self, bgr_frame: np.ndarray) -> list[dict]:
        """
        Detect & identify every face in a BGR frame.

        Returns a list of dicts:
          bbox        : (x1, y1, x2, y2)  — int pixel coords
          name        : str               — identity or "Unknown"
          distance    : float             — cosine distance (lower = closer)
          confidence  : float             — MTCNN detector confidence
          is_auth     : bool
          face_rgb    : np.ndarray        — cropped RGB face image (raw size)
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        results: list[dict] = []

        try:
            faces = self.detector.detect_faces(rgb)
        except Exception as exc:
            print(f"[FaceAuth] Detection error: {exc}")
            return results

        if not faces:
            return results

        for face in faces:
            conf = face.get("confidence", 1.0)
            if conf < self.confidence_threshold:
                continue

            x, y, w, h = face["box"]
            x, y = max(0, x), max(0, y)
            w, h = max(0, w), max(0, h)
            x2 = min(x + w, rgb.shape[1])
            y2 = min(y + h, rgb.shape[0])

            if (x2 - x) < self.min_face_size or (y2 - y) < self.min_face_size:
                continue

            face_crop = rgb[y:y2, x:x2]
            if face_crop.size == 0:
                continue

            face_resized = cv2.resize(face_crop, (160, 160))

            try:
                emb = self.embedder.embeddings([face_resized])[0]
            except Exception as exc:
                print(f"[FaceAuth] Embedding error: {exc}")
                continue

            distances = np.array(
                [cosine(emb, known) for known in self.known_embeddings]
            )
            min_dist = float(distances.min())
            best_idx = int(distances.argmin())

            if min_dist < self.threshold:
                name = str(self.known_names[best_idx])
                is_auth = True
            else:
                name = "Unknown"
                is_auth = False

            results.append(
                {
                    "bbox": (x, y, x2, y2),
                    "name": name,
                    "distance": min_dist,
                    "confidence": conf,
                    "is_auth": is_auth,
                    "face_rgb": face_crop,
                }
            )

        return results

    # ------------------------------------------------------------------
    def draw_results(self, bgr_frame: np.ndarray, auth_results: list[dict]) -> np.ndarray:
        """Overlay bounding boxes + labels on a copy of the frame."""
        out = bgr_frame.copy()
        for r in auth_results:
            x1, y1, x2, y2 = r["bbox"]
            color = (0, 220, 0) if r["is_auth"] else (0, 0, 220)
            status = f"{'AUTH' if r['is_auth'] else 'UNAUTH'}: {r['name']}"

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Label background
            (lw, lh), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(out, (x1, y1 - lh - 14), (x1 + lw + 6, y1), color, -1)
            cv2.putText(
                out, status, (x1 + 3, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
            )

            # Distance sub-label
            dist_txt = f"d={r['distance']:.3f}  c={r['confidence']:.2f}"
            cv2.putText(
                out, dist_txt, (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
            )

        return out