from __future__ import annotations

import os
import cv2
import numpy as np


def detect_faces(image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = os.path.join(cv2.__path__[0], "data", "haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade_path):
        return []

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        return []

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(80, 80),
    )
    return [tuple(map(int, f)) for f in faces]


def detect_face_mask(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(image_bgr)
    mask = np.zeros(gray.shape, dtype=np.uint8)

    if len(faces) == 0:
        return mask

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    x1 = max(0, x - int(w * 0.35))
    y1 = max(0, y - int(h * 0.35))
    x2 = min(gray.shape[1], x + w + int(w * 0.35))
    y2 = min(gray.shape[0], y + h + int(h * 0.85))

    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    mask = cv2.GaussianBlur(mask, (41, 41), 0)
    return mask
