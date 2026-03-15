from __future__ import annotations

import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent.parent / "store.db"

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

DEFAULT_INVENTORY = {
    "Tesco Diet Lemonade": 10,
    "Schweppes Slimline Lemonade": 16,
    "Robinsons Fruit Creations Lime Cordial": 8,
    "Bottlegreen Elderflower Cordial": 4,
    "Kellogg's Corn Flakes": 9,
    "Weetabix Original": 7,
    "Heinz Baked Beans": 18,
    "Barilla Spaghetti": 14,
    "Walkers Ready Salted Crisps": 20,
    "Fairy Washing Up Liquid": 11,
    "Colgate Total Toothpaste": 13,
    "Andrex Toilet Tissue": 12,
    "Ocean Spray Cranberry": 6,
    "Capri-Sun Zero Orange": 13,
    "Fruit Shoot Apple & Blackcurrant": 15
}


def init_db() -> None:
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS inventory(
            product TEXT PRIMARY KEY,
            backstock INTEGER NOT NULL DEFAULT 0
        )
        """
    )

    for product, backstock in DEFAULT_INVENTORY.items():
        cursor.execute(
            """
            INSERT INTO inventory(product, backstock)
            VALUES(?, ?)
            ON CONFLICT(product) DO NOTHING
            """,
            (product, backstock),
        )

    conn.commit()
