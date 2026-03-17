from __future__ import annotations

import numpy as np

from pencil_sketch_app.config.settings import (
    MODE_AUTO_AI,
    MODE_INSTANTID,
    MODE_IPADAPTER,
    MODE_LOCAL,
    MODE_OPENAI,
)
from pencil_sketch_app.core.face_detection import detect_faces
from pencil_sketch_app.core.line_art import make_pencil_line_art
from pencil_sketch_app.pipelines.instantid_pipeline import generate_with_instantid
from pencil_sketch_app.pipelines.ipadapter_pipeline import generate_with_ipadapter
from pencil_sketch_app.pipelines.openai_pipeline import openai_edit_to_pencil


def pick_reference_mode(image_bgr: np.ndarray) -> str:
    faces = detect_faces(image_bgr)
    if len(faces) >= 1:
        return MODE_INSTANTID
    return MODE_IPADAPTER


def process_image_by_mode(
    image_bgr: np.ndarray,
    mode: str,
    prompt: str,
    openai_api_key: str,
    openai_model: str,
    openai_quality: str,
    openai_size: str,
    similarity_strength: int,
    low_memory_mode: bool,
    line_art_settings: dict,
) -> tuple[np.ndarray, str]:
    if mode == MODE_OPENAI:
        result = openai_edit_to_pencil(
            image_bgr,
            api_key=openai_api_key,
            prompt=prompt,
            model=openai_model,
            quality=openai_quality,
            size=openai_size,
        )
        return result, MODE_OPENAI

    if mode == MODE_LOCAL:
        result = make_pencil_line_art(image_bgr, **line_art_settings)
        return result, MODE_LOCAL

    effective_mode = mode
    if mode == MODE_AUTO_AI:
        effective_mode = pick_reference_mode(image_bgr)

    if effective_mode == MODE_INSTANTID:
        result = generate_with_instantid(
            image_bgr=image_bgr,
            prompt=prompt,
            similarity_strength=similarity_strength,
            low_memory_mode=low_memory_mode,
        )
        return result, effective_mode

    if effective_mode == MODE_IPADAPTER:
        result = generate_with_ipadapter(
            image_bgr=image_bgr,
            prompt=prompt,
            similarity_strength=similarity_strength,
            low_memory_mode=low_memory_mode,
        )
        return result, effective_mode

    raise RuntimeError(f"Неизвестный режим обработки: {mode}")
