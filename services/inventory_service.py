from __future__ import annotations

import re

from database.db import conn, cursor


def _normalize_product_name(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def check_backstock(product: str) -> int:
    cursor.execute("SELECT backstock FROM inventory WHERE product = ?", (product,))
    row = cursor.fetchone()
    if row is not None:
        return int(row["backstock"])

    normalized_target = _normalize_product_name(product)
    cursor.execute("SELECT product, backstock FROM inventory")

    best_match: int | None = None
    best_score = 0
    target_tokens = set(normalized_target.split())

    for candidate in cursor.fetchall():
        candidate_name = candidate["product"]
        candidate_tokens = set(_normalize_product_name(candidate_name).split())
        score = len(target_tokens & candidate_tokens)

        if score > best_score and score >= 2:
            best_score = score
            best_match = int(candidate["backstock"])

    return best_match or 0


def list_inventory() -> list[dict[str, int | str]]:
    cursor.execute("SELECT product, backstock FROM inventory ORDER BY product")
    return [dict(row) for row in cursor.fetchall()]


def upsert_inventory(product: str, backstock: int) -> None:
    cursor.execute(
        """
        INSERT INTO inventory(product, backstock)
        VALUES(?, ?)
        ON CONFLICT(product) DO UPDATE SET backstock = excluded.backstock
        """,
        (product, backstock),
    )
    conn.commit()
