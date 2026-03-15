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
