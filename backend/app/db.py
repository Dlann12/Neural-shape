"""SQLite persistence for projects, jobs and ML experiments."""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT DEFAULT 'chair',
        status TEXT NOT NULL,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        validation_json TEXT,
        results_json TEXT,
        metrics_json TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        project_id TEXT,
        job_type TEXT NOT NULL,
        method TEXT NOT NULL,
        status TEXT NOT NULL,
        progress INTEGER DEFAULT 0,
        error TEXT,
        created_at REAL NOT NULL,
        duration_ms REAL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ml_experiments (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        status TEXT NOT NULL,
        config_json TEXT NOT NULL,
        result_json TEXT,
        progress_json TEXT,
        created_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS action_figures (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        status TEXT NOT NULL,
        mc_resolution INTEGER DEFAULT 128,
        progress INTEGER DEFAULT 0,
        progress_msg TEXT,
        error TEXT,
        vertex_count INTEGER DEFAULT 0,
        face_count INTEGER DEFAULT 0,
        duration_seconds REAL DEFAULT 0,
        metadata_json TEXT,
        created_at REAL NOT NULL,
        completed_at REAL
    )
    """,
]


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            for stmt in SCHEMA:
                conn.execute(stmt)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(ml_experiments)")}
            if "progress_json" not in cols:  # migrate databases created by the first MVP version
                conn.execute("ALTER TABLE ml_experiments ADD COLUMN progress_json TEXT")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def execute(self, sql: str, params: tuple = ()) -> None:
        with self.connect() as conn:
            conn.execute(sql, params)

    def fetch_one(self, sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def fetch_all(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]


def loads(value: Optional[str]) -> Any:
    return json.loads(value) if value else None
