from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATABASE_PATH = Path(__file__).with_name("simulations.db")


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS simulations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                icon TEXT NOT NULL,
                hormone TEXT NOT NULL,
                profile_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def list_simulations(limit: int = 50) -> list[dict[str, Any]]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM simulations ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [
        {
            "id": row["id"],
            "type": row["simulation_type"],
            "title": row["title"],
            "description": row["description"],
            "icon": row["icon"],
            "hormone": row["hormone"],
            "date": row["created_at"],
            "profile": json.loads(row["profile_json"]),
        }
        for row in rows
    ]


def create_simulation(simulation: dict[str, Any]) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO simulations
                (simulation_type, title, description, icon, hormone, profile_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                simulation.get("type", "scenario"),
                simulation["title"],
                simulation["description"],
                simulation["icon"],
                simulation["hormone"],
                json.dumps(simulation["profile"]),
                created_at,
            ),
        )
        simulation_id = cursor.lastrowid
    return {**simulation, "id": simulation_id, "date": created_at}


def delete_simulation(simulation_id: int) -> bool:
    with _connect() as connection:
        cursor = connection.execute("DELETE FROM simulations WHERE id = ?", (simulation_id,))
    return cursor.rowcount > 0
