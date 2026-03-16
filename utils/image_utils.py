from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageOps

try:
    import cv2
except ImportError:  # pragma: no cover - optional during first bootstrap
    cv2 = None


def preprocess_image(image_bytes: bytes) -> tuple[Image.Image, bytes, str]:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = ImageOps.exif_transpose(image)
    image.thumbnail((1024, 1024))

    canvas = Image.new("RGB", (1024, 1024), color=(248, 248, 244))
    offset = ((1024 - image.width) // 2, (1024 - image.height) // 2)
    canvas.paste(image, offset)

    output = BytesIO()
    canvas.save(output, format="JPEG", quality=92)
    return canvas, output.getvalue(), "jpeg"


def summarize_image(image: Image.Image) -> dict[str, Any]:
    image_array = np.array(image)
    gray = np.mean(image_array, axis=2).astype(np.uint8)

    brightness = float(np.mean(gray))

    if cv2 is not None:
        edges = cv2.Canny(gray, 80, 140)
        edge_density = float(np.count_nonzero(edges) / edges.size)
    else:
        gradient_x = np.abs(np.diff(gray.astype(np.int16), axis=1))
        edge_density = float(np.count_nonzero(gradient_x > 16) / gradient_x.size)

    return {
        "brightness": brightness,
        "edge_density": edge_density,
    }


def create_row_crops(image: Image.Image, row_count: int) -> list[tuple[int, bytes, str]]:
    if row_count <= 0:
        return []

    width, height = image.size
    left = int(width * 0.08)
    right = int(width * 0.92)
    top = int(height * 0.04)
    bottom = int(height * 0.92)
    cropped_height = max(1, bottom - top)
    row_height = max(1, cropped_height // row_count)
    overlap = max(12, int(row_height * 0.14))

    crops: list[tuple[int, bytes, str]] = []
    for index in range(row_count):
        y1 = max(0, top + index * row_height - overlap)
        y2 = min(height, top + (index + 1) * row_height + overlap)
        row_image = image.crop((left, y1, right, y2))
        output = BytesIO()
        row_image.save(output, format="JPEG", quality=92)
        crops.append((index + 1, output.getvalue(), "jpeg"))

    return crops



def create_row_crop_images(image: Image.Image, row_count: int) -> list[tuple[int, Image.Image]]:
    if row_count <= 0:
        return []

    width, height = image.size
    left = int(width * 0.08)
    right = int(width * 0.92)
    top = int(height * 0.04)
    bottom = int(height * 0.92)
    cropped_height = max(1, bottom - top)
    row_height = max(1, cropped_height // row_count)
    overlap = max(12, int(row_height * 0.14))

    row_images: list[tuple[int, Image.Image]] = []
    for index in range(row_count):
        y1 = max(0, top + index * row_height - overlap)
        y2 = min(height, top + (index + 1) * row_height + overlap)
        row_images.append((index + 1, image.crop((left, y1, right, y2))))

    return row_images


def detect_gap_segments(row_image: Image.Image) -> list[dict[str, float | int | str]]:
    image_array = np.asarray(row_image.convert('RGB'))
    if image_array.size == 0:
        return []

    height, width, _ = image_array.shape
    y1 = int(height * 0.12)
    y2 = int(height * 0.82)
    focus = image_array[y1:y2, :, :]
    if focus.size == 0:
        focus = image_array

    gray = np.mean(focus, axis=2).astype(np.uint8)
    brightness = gray.mean(axis=0)
    white_ratio = (gray > 185).mean(axis=0)
    texture = gray.std(axis=0)

    if cv2 is not None:
        edges = cv2.Canny(gray, 80, 150)
        edge_strength = edges.mean(axis=0) / 255.0
    else:
        gradient = np.abs(np.diff(gray.astype(np.int16), axis=1))
        edge_strength = gradient.mean(axis=0)
        if edge_strength.max() > 0:
            edge_strength = edge_strength / edge_strength.max()

    texture_norm = texture / texture.max() if texture.max() > 0 else texture
    brightness_norm = brightness / 255.0
    activity = (0.55 * edge_strength) + (0.35 * texture_norm) + (0.10 * (1.0 - brightness_norm))

    window = max(9, (width // 28) | 1)
    kernel = np.ones(window, dtype=np.float32) / window
    smooth_activity = np.convolve(activity, kernel, mode='same')
    smooth_white = np.convolve(white_ratio, kernel, mode='same')
    smooth_brightness = np.convolve(brightness_norm, kernel, mode='same')

    low_activity_threshold = min(0.22, float(np.percentile(smooth_activity, 30)) + 0.04)
    min_width = max(18, int(width * 0.045))
    max_width = max(min_width + 1, int(width * 0.22))
    margin = int(width * 0.05)

    segments: list[dict[str, float | int | str]] = []
    start: int | None = None
    for idx, value in enumerate(smooth_activity):
        is_gap_like = value <= low_activity_threshold and smooth_white[idx] >= 0.16 and smooth_brightness[idx] >= 0.48
        if is_gap_like and start is None:
            start = idx
        elif not is_gap_like and start is not None:
            end = idx
            width_px = end - start
            if margin <= start and end <= width - margin and min_width <= width_px <= max_width:
                center_ratio = ((start + end) / 2.0) / max(width, 1)
                if center_ratio < 0.33:
                    location = 'left'
                elif center_ratio > 0.66:
                    location = 'right'
                else:
                    location = 'center'
                confidence = min(0.9, max(0.45, (smooth_white[start:end].mean() * 0.6) + ((1.0 - smooth_activity[start:end].mean()) * 0.4)))
                estimated_missing_units = max(1, round(width_px / max(1, int(width * 0.075))))
                segments.append(
                    {
                        'start_px': start,
                        'end_px': end,
                        'location_on_shelf': location,
                        'confidence': round(float(confidence), 3),
                        'estimated_missing_units': int(estimated_missing_units),
                    }
                )
            start = None

    if start is not None:
        end = width
        width_px = end - start
        if margin <= start and end <= width - margin and min_width <= width_px <= max_width:
            center_ratio = ((start + end) / 2.0) / max(width, 1)
            location = 'left' if center_ratio < 0.33 else 'right' if center_ratio > 0.66 else 'center'
            confidence = min(0.9, max(0.45, (smooth_white[start:end].mean() * 0.6) + ((1.0 - smooth_activity[start:end].mean()) * 0.4)))
            estimated_missing_units = max(1, round(width_px / max(1, int(width * 0.075))))
            segments.append(
                {
                    'start_px': start,
                    'end_px': end,
                    'location_on_shelf': location,
                    'confidence': round(float(confidence), 3),
                    'estimated_missing_units': int(estimated_missing_units),
                }
            )

    return segments[:4]
