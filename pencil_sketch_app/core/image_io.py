from __future__ import annotations

import cv2
import numpy as np
from io import BytesIO
from pathlib import Path
from PIL import Image


def safe_filename(name: str) -> str:
    forbidden = '<>:"/\\|?*'
    cleaned = "".join("_" if c in forbidden else c for c in name)
    cleaned = cleaned.strip().strip(".")
    return cleaned or "image"


def cv_read_image_unicode(file_path: str | Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(file_path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def cv_write_image_unicode(file_path: str | Path, image: np.ndarray) -> bool:
    try:
        ext = Path(file_path).suffix.lower()
        if not ext:
            ext = ".png"
            file_path = str(file_path) + ext
        ok, buffer = cv2.imencode(ext, image)
        if not ok:
            return False
        buffer.tofile(str(file_path))
        return True
    except Exception:
        return False


def image_bgr_to_png_bytes(image_bgr: np.ndarray) -> bytes:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    bio = BytesIO()
    pil_img.save(bio, format="PNG")
    return bio.getvalue()


def decode_b64_image_to_bgr(b64_data: str) -> np.ndarray:
    import base64

    raw = base64.b64decode(b64_data)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Не удалось декодировать изображение, полученное от API.")
    return img


def auto_resize(image: np.ndarray, max_side: int = 1800) -> np.ndarray:
    h, w = image.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return image
    scale = max_side / side
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
