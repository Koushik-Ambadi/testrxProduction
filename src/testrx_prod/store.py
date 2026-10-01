from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class InvocationStore:
    """Normalized application audit log; platform telemetry joins on invocation_id."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.getenv("TESTRX_LOG_DB", "/data/testrx.sqlite3"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS invocation (
                    invocation_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                    query TEXT NOT NULL, config_id TEXT NOT NULL,
                    index_release_id TEXT NOT NULL, candidate_k INTEGER NOT NULL,
                    top_k INTEGER NOT NULL, encoder_id TEXT NOT NULL,
                    reranker_id TEXT NOT NULL, answer TEXT
                );
                CREATE TABLE IF NOT EXISTS prompt_message (
                    invocation_id TEXT NOT NULL REFERENCES invocation(invocation_id),
                    ordinal INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
                    PRIMARY KEY (invocation_id, ordinal)
                );
                CREATE TABLE IF NOT EXISTS ranking (
                    invocation_id TEXT NOT NULL REFERENCES invocation(invocation_id),
                    chunk_id TEXT NOT NULL, dense_rank INTEGER NOT NULL,
                    dense_score REAL NOT NULL, rerank_rank INTEGER NOT NULL,
                    rerank_score REAL NOT NULL, selected_rank INTEGER,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (invocation_id, chunk_id)
                );
                CREATE INDEX IF NOT EXISTS ranking_by_chunk ON ranking(chunk_id);
            """)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def record(self, invocation: dict[str, Any], rankings: list[dict[str, Any]]) -> None:
        with self.connect() as db:
            db.execute("""INSERT INTO invocation VALUES
                (:invocation_id,:created_at,:query,:config_id,:index_release_id,
                 :candidate_k,:top_k,:encoder_id,:reranker_id,NULL)""", invocation)
            db.executemany("INSERT INTO prompt_message VALUES (?,?,?,?)", [
                (invocation["invocation_id"], ordinal, message["role"], message["content"])
                for ordinal, message in enumerate(invocation["prompt"])
            ])
            db.executemany("""INSERT INTO ranking VALUES
                (:invocation_id,:chunk_id,:dense_rank,:dense_score,:rerank_rank,
                 :rerank_score,:selected_rank,:payload_json)""", [
                {**row, "payload_json": json.dumps(row["payload"], ensure_ascii=False, separators=(",", ":"))}
                for row in rankings
            ])

    def record_answer(self, invocation_id: str, answer: str) -> None:
        with self.connect() as db:
            cursor = db.execute("UPDATE invocation SET answer=? WHERE invocation_id=?", (answer, invocation_id))
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown invocation_id: {invocation_id}")
