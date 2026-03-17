from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def resize_for_generation(image_bgr: np.ndarray, target_long_side: int = 1024) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    long_side = max(h, w)
    if long_side <= target_long_side:
        src = image_bgr.copy()
    else:
        scale = target_long_side / long_side
        new_w = max(64, int(round(w * scale / 64) * 64))
        new_h = max(64, int(round(h * scale / 64) * 64))
        src = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    h2, w2 = src.shape[:2]
    if w2 % 64 != 0 or h2 % 64 != 0:
        w2 = max(64, int(round(w2 / 64) * 64))
        h2 = max(64, int(round(h2 / 64) * 64))
        src = cv2.resize(src, (w2, h2), interpolation=cv2.INTER_AREA)
    return src


def bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_bgr(image_pil: Image.Image) -> np.ndarray:
    rgb = np.array(image_pil.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def make_control_image(image_bgr: np.ndarray, target_long_side: int = 1024) -> Image.Image:
    resized = resize_for_generation(image_bgr, target_long_side=target_long_side)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 50, 50)
    gray = cv2.bilateralFilter(gray, 9, 60, 60)
    edges = cv2.Canny(gray, 60, 140)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    control = 255 - edges
    control_rgb = cv2.cvtColor(control, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(control_rgb)


def make_face_keypoint_image(image_bgr: np.ndarray, keypoints: np.ndarray, target_long_side: int = 1024) -> Image.Image:
    resized = resize_for_generation(image_bgr, target_long_side=target_long_side)
    h, w = resized.shape[:2]
    out_img = np.zeros((h, w, 3), dtype=np.uint8)

    kps = np.array(keypoints, dtype=np.float32)
    if kps.shape != (5, 2):
        raise RuntimeError("InstantID ожидает 5 ключевых точек лица.")

    color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    limb_seq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    stick_width = 4

    for index in limb_seq:
        color = color_list[index[0]]
        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = float(((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5)
        angle = float(np.degrees(np.arctan2(y[0] - y[1], x[0] - x[1])))
        polygon = cv2.ellipse2Poly(
            (int(np.mean(x)), int(np.mean(y))),
            (int(length / 2), stick_width),
            int(angle),
            0,
            360,
            1,
        )
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
        out_img = (out_img * 0.6).astype(np.uint8)

    for idx, kp in enumerate(kps):
        color = color_list[idx]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    return Image.fromarray(out_img)


def prepare_reference_image(image_bgr: np.ndarray, target_long_side: int = 1024) -> Image.Image:
    resized = resize_for_generation(image_bgr, target_long_side=target_long_side)
    return bgr_to_pil(resized)
