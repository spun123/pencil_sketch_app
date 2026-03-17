from __future__ import annotations

from io import BytesIO
import numpy as np

from pencil_sketch_app.config.settings import DEFAULT_OPENAI_MODEL
from pencil_sketch_app.core.image_io import image_bgr_to_png_bytes, decode_b64_image_to_bgr

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def openai_edit_to_pencil(
    image_bgr: np.ndarray,
    api_key: str,
    prompt: str,
    model: str = DEFAULT_OPENAI_MODEL,
    quality: str = "high",
    size: str = "1024x1024",
) -> np.ndarray:
    if OpenAI is None:
        raise RuntimeError("Библиотека openai не установлена. Установите её командой: pip install openai")

    api_key = api_key.strip()
    if not api_key:
        raise RuntimeError("Не указан API-ключ OpenAI.")

    png_bytes = image_bgr_to_png_bytes(image_bgr)
    client = OpenAI(api_key=api_key)

    image_file = BytesIO(png_bytes)
    image_file.name = "input.png"

    response = client.images.edit(
        model=model,
        image=image_file,
        prompt=prompt,
        quality=quality,
        size=size,
    )

    if not response.data or not getattr(response.data[0], "b64_json", None):
        raise RuntimeError("API не вернуло итоговое изображение.")

    return decode_b64_image_to_bgr(response.data[0].b64_json)
