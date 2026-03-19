"""
object_detect.py — Real-time Object Detection
Supports YOLOv11 and YOLOv12 via the `ultralytics` package.

Install:  pip install ultralytics
"""

from __future__ import annotations

import cv2
import numpy as np

# Colour palette — distinct hue per class_id (mod 20)
_PALETTE = [
    (255, 165,   0),   # orange
    ( 52, 152, 219),   # blue
    ( 46, 204, 113),   # green
    (231,  76,  60),   # red
    (155,  89, 182),   # purple
    ( 26, 188, 156),   # teal
    (241, 196,  15),   # yellow
    (230, 126,  34),   # dark-orange
    (149, 165, 166),   # grey
    ( 52,  73,  94),   # dark-grey
    (236, 240, 241),   # light
    (  0, 188, 212),   # cyan
    (233,  30,  99),   # pink
    (121,  85, 72),    # brown
    ( 63,  81, 181),   # indigo
    ( 76, 175,  80),   # light-green
    (255, 235,  59),   # lime
    (103,  58, 183),   # deep-purple
    (  0, 150, 136),   # dark-teal
    (255, 152,   0),   # amber
]


class ObjectDetector:
    """
    Thin wrapper around an Ultralytics YOLO model.

    Parameters
    ----------
    model_name : str
        e.g. "yolo11n.pt", "yolo11s.pt", "yolo12n.pt"
        The model is auto-downloaded on first use.
    conf_threshold : float
        Minimum detection confidence to keep.
    iou_threshold : float
        NMS IoU threshold.
    device : str | None
        "cpu", "cuda", "mps", or None for auto-select.
    """

    SUPPORTED_MODELS = [
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
        "yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt",
    ]

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        device: str | None = None,
    ):
        from ultralytics import YOLO

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = YOLO(model_name)
        self.device = device
        print(f"[ObjectDetector] Loaded '{model_name}'.")

    # ------------------------------------------------------------------
    def detect(self, bgr_frame: np.ndarray) -> tuple[list[dict], np.ndarray]:
        """
        Run detection on a BGR frame.

        Returns
        -------
        objects : list[dict]
            Each dict has keys:
              name       : str    — class label
              confidence : float  — detection score
              bbox       : (x1, y1, x2, y2) int pixel coords
              class_id   : int
        annotated : np.ndarray
            BGR frame with bounding boxes drawn.
        """
        kwargs = dict(
            verbose=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
        )
        if self.device:
            kwargs["device"] = self.device

        results = self.model(bgr_frame, **kwargs)

        objects: list[dict] = []
        annotated = bgr_frame.copy()

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                name = self.model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                objects.append(
                    {
                        "name": name,
                        "confidence": conf,
                        "bbox": (x1, y1, x2, y2),
                        "class_id": cls_id,
                    }
                )

                color = _PALETTE[cls_id % len(_PALETTE)]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                label = f"{name} {conf:.2f}"
                (lw, lh), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                )
                cv2.rectangle(
                    annotated, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1
                )
                cv2.putText(
                    annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                )

        return objects, annotated

    # ------------------------------------------------------------------
    @staticmethod
    def summarise(objects: list[dict]) -> dict[str, int]:
        """Return {class_name: count} from a detection list."""
        summary: dict[str, int] = {}
        for o in objects:
            summary[o["name"]] = summary.get(o["name"], 0) + 1
        return summary