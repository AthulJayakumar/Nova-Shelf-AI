from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from database.models import PlanogramItem


PLANOGRAM_PATH = Path(__file__).resolve().parent.parent / "data" / "planogram.json"
DEFAULT_LOCATION = "Unknown aisle"


@lru_cache
def get_planogram() -> dict[str, PlanogramItem]:
    raw_data = json.loads(PLANOGRAM_PATH.read_text(encoding="utf-8-sig"))
    return {name: PlanogramItem(**payload) for name, payload in raw_data.items()}


def get_planogram_item(product: str | None) -> PlanogramItem | None:
    if not product:
        return None
    return get_planogram().get(product)


def resolve_location(product: str | None, fallback_location: str, fallback_shelf_level: str) -> tuple[str, str]:
    item = get_planogram_item(product)
    if item is not None:
        return item.location, item.shelf_level
    return fallback_location or DEFAULT_LOCATION, fallback_shelf_level
