from __future__ import annotations

from database.models import ShelfAssessment


def analyze_gap(detected_units: int, expected_units: int) -> ShelfAssessment:
    gap = max(0, expected_units - detected_units)

    if detected_units == 0:
        status = "OUT_OF_STOCK"
    elif detected_units < expected_units:
        status = "LOW_STOCK"
    else:
        status = "OK"

    return ShelfAssessment(
        status=status,
        gap=gap,
        expected_units=expected_units,
        detected_units=detected_units,
    )
