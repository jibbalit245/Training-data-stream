"""
db_indexer.py
SQLite-based index for the extraction pipeline.

Tables
------
documents
    doc_id, tier, structural_prior, domain, doc_type, quality_score,
    source_url, agent_id, extraction_timestamp, license

semantic_nodes
    fingerprint (SHA-256), source_doc_id, source_url

extraction_log
    id (auto), agent_id, source, status, records_extracted,
    error_message, started_at, finished_at
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DB_PATH = os.getenv("DB_PATH", "pipeline.db")

_DDL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id              TEXT PRIMARY KEY,
    tier                INTEGER,
    structural_prior    INTEGER,
    domain              TEXT,       -- JSON list
    doc_type            TEXT,
    quality_score       REAL,
    source_url          TEXT,
    agent_id            TEXT,
    extraction_timestamp TEXT,
    license             TEXT
);

CREATE TABLE IF NOT EXISTS semantic_nodes (
    fingerprint     TEXT PRIMARY KEY,
    source_doc_id   TEXT,
    source_url      TEXT,
    FOREIGN KEY (source_doc_id) REFERENCES documents(doc_id)
);

CREATE TABLE IF NOT EXISTS extraction_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id            TEXT,
    source              TEXT,
    status              TEXT,       -- started | completed | failed
    records_extracted   INTEGER DEFAULT 0,
    error_message       TEXT,
    started_at          TEXT,
    finished_at         TEXT
);

CREATE INDEX IF NOT EXISTS idx_doc_tier ON documents(tier);
CREATE INDEX IF NOT EXISTS idx_doc_type ON documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_log_agent ON extraction_log(agent_id);
"""


class DBIndexer:
    """
    Thread-safe SQLite indexer.

    Parameters
    ----------
    db_path : str
        Path to the SQLite file (created automatically).
    """

    def __init__(self, db_path: str = _DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._get_conn() as conn:
                conn.executescript(_DDL)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def index_record(self, record: dict) -> None:
        """Index a single extracted record."""
        import json
        meta = record.get("metadata", {})
        doc_id = record.get("doc_id", "")
        fingerprint = hashlib.sha256(
            record.get("text", "").encode("utf-8", errors="replace")
        ).hexdigest()

        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO documents
                        (doc_id, tier, structural_prior, domain, doc_type,
                         quality_score, source_url, agent_id,
                         extraction_timestamp, license)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        meta.get("tier"),
                        meta.get("structural_prior"),
                        json.dumps(meta.get("domain", [])),
                        meta.get("doc_type"),
                        meta.get("quality_score"),
                        meta.get("source_url"),
                        meta.get("agent_id"),
                        meta.get("extraction_timestamp"),
                        meta.get("license"),
                    ),
                )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO semantic_nodes
                        (fingerprint, source_doc_id, source_url)
                    VALUES (?, ?, ?)
                    """,
                    (fingerprint, doc_id, meta.get("source_url", "")),
                )

    def log_start(self, agent_id: str, source: str) -> int:
        """Log the start of an extraction run. Returns the log row id."""
        with self._lock:
            with self._get_conn() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO extraction_log
                        (agent_id, source, status, started_at)
                    VALUES (?, ?, 'started', ?)
                    """,
                    (agent_id, source, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
                )
                return cur.lastrowid  # type: ignore[return-value]

    def log_finish(
        self,
        log_id: int,
        status: str = "completed",
        records_extracted: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    UPDATE extraction_log
                    SET status=?, records_extracted=?, error_message=?, finished_at=?
                    WHERE id=?
                    """,
                    (
                        status,
                        records_extracted,
                        error_message,
                        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        log_id,
                    ),
                )

    def summary(self) -> Dict[str, Any]:
        """Return a dict with high-level pipeline statistics."""
        with self._lock:
            with self._get_conn() as conn:
                total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                by_tier = dict(
                    conn.execute(
                        "SELECT tier, COUNT(*) FROM documents GROUP BY tier"
                    ).fetchall()
                )
                by_type = dict(
                    conn.execute(
                        "SELECT doc_type, COUNT(*) FROM documents GROUP BY doc_type"
                    ).fetchall()
                )
                log_rows = conn.execute(
                    "SELECT status, COUNT(*) FROM extraction_log GROUP BY status"
                ).fetchall()
        return {
            "total_documents": total,
            "by_tier": by_tier,
            "by_doc_type": by_type,
            "extraction_log": dict(log_rows),
        }
