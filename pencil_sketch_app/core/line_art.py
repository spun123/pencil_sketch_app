from __future__ import annotations

import cv2
import numpy as np

from .image_io import auto_resize
from .face_detection import detect_face_mask


def remove_small_components(binary_img: np.ndarray, min_area: int = 14) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    cleaned = np.zeros_like(binary_img)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def simplify_for_line_art(gray: np.ndarray) -> np.ndarray:
    base = cv2.bilateralFilter(gray, 9, 50, 50)
    base = cv2.bilateralFilter(base, 9, 60, 60)
    base = cv2.medianBlur(base, 5)
    return base


def xdog_edges(gray: np.ndarray) -> np.ndarray:
    g1 = cv2.GaussianBlur(gray, (0, 0), 0.8)
    g2 = cv2.GaussianBlur(gray, (0, 0), 1.6)

    g1 = g1.astype(np.float32) / 255.0
    g2 = g2.astype(np.float32) / 255.0

    dog = g1 - 0.98 * g2
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, edges = cv2.threshold(dog, 140, 255, cv2.THRESH_BINARY)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return edges


def build_clean_line_map(
    image_bgr: np.ndarray,
    contour_low: int,
    contour_high: int,
    noise_cleaning: int,
) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_bg = cv2.GaussianBlur(gray, (31, 31), 0)
    normalized = cv2.divide(gray, blur_bg, scale=255)
    simplified = simplify_for_line_art(normalized)

    canny = cv2.Canny(simplified, contour_low, contour_high)
    xdog = xdog_edges(simplified)

    combined = cv2.bitwise_or(canny, xdog)
    combined = remove_small_components(combined, min_area=noise_cleaning)
    return combined


def make_pencil_line_art(
    image_bgr: np.ndarray,
    contour_low: int = 60,
    contour_high: int = 140,
    line_brightness: int = 168,
    noise_cleaning: int = 28,
    line_thickness: int = 1,
    keep_extra_details: bool = False,
) -> np.ndarray:
    image_bgr = auto_resize(image_bgr, max_side=1800)

    line_map = build_clean_line_map(
        image_bgr,
        contour_low=contour_low,
        contour_high=contour_high,
        noise_cleaning=noise_cleaning,
    )

    line_ratio = np.count_nonzero(line_map) / max(1, line_map.size)
    if line_ratio > 0.35:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        simple = simplify_for_line_art(gray)
        line_map = cv2.Canny(simple, contour_low, contour_high)

    face_mask = detect_face_mask(image_bgr)
    if np.count_nonzero(face_mask) > 0:
        face_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        face_simple = simplify_for_line_art(face_gray)
        face_edges = cv2.Canny(face_simple, max(20, contour_low + 10), max(70, contour_high + 25))
        face_edges = remove_small_components(face_edges, min_area=max(18, noise_cleaning + 6))

        inv_mask = cv2.bitwise_not(face_mask)
        outside = cv2.bitwise_and(line_map, line_map, mask=inv_mask)
        inside = cv2.bitwise_and(face_edges, face_edges, mask=face_mask)
        line_map = cv2.bitwise_or(outside, inside)

    if keep_extra_details:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        soft = cv2.adaptiveThreshold(
            simplify_for_line_art(gray),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            25,
            10,
        )
        soft = cv2.morphologyEx(soft, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        soft = remove_small_components(soft, min_area=max(25, noise_cleaning + 8))
        line_map = cv2.bitwise_or(line_map, soft)

    if line_thickness > 1:
        kernel = np.ones((line_thickness, line_thickness), np.uint8)
        line_map = cv2.dilate(line_map, kernel, iterations=1)

    result = np.full(line_map.shape, 255, dtype=np.uint8)
    result[line_map > 0] = line_brightness
    result = cv2.GaussianBlur(result, (3, 3), 0)
    result[result > 245] = 255
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
