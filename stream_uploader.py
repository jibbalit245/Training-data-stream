"""
stream_uploader.py
Local Parquet dataset writer.

Design
------
- Buffers extraction records in memory.
- Flushes when buffer reaches ``chunk_size_mb`` MB (uncompressed JSON estimate).
- Writes each chunk as a Parquet shard to a local output directory located
  one level above the repository root (sibling of the repo, not inside it).
- Thread-safe: multiple extractor threads can call ``add()`` concurrently.
- Provides a progress summary before discarding buffer.
- No network uploads: everything is stored on local disk.

Output location
---------------
Chunks are written to ``<output_dir>/<split>/chunk_XXXXXX.parquet``.
The default ``output_dir`` is the ``output`` folder immediately above the
repository (i.e. ``../output`` relative to this file's directory), so the
data lives outside the repo tree.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from threading import Lock
from typing import List, Optional

logger = logging.getLogger(__name__)

_BYTES_PER_MB = 1024 * 1024
_CHUNK_SIZE_MB = float(os.getenv("HF_CHUNK_SIZE_MB", "50"))
_HF_SPLIT = os.getenv("HF_SPLIT", "train")

# Default output directory: one level above the repo root
_DEFAULT_OUTPUT_DIR = str(
    Path(__file__).resolve().parent.parent / "output"
)
_OUTPUT_DIR = os.getenv("OUTPUT_DIR", _DEFAULT_OUTPUT_DIR)


def _json_bytes(record: dict) -> int:
    return len(json.dumps(record, ensure_ascii=False).encode("utf-8")) + 1


def _write_chunk_local(
    records: List[dict],
    output_dir: str,
    chunk_index: int,
    split: str = "train",
) -> None:
    """
    Serialise *records* to Parquet and write to *output_dir*/<split>/.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not records:
        return

    dest_dir = Path(output_dir) / split
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_path = dest_dir / f"chunk_{chunk_index:06d}.parquet"

    table = pa.Table.from_pylist(records)
    pq.write_table(table, str(dest_path), compression="snappy")

    size_mb = dest_path.stat().st_size / _BYTES_PER_MB
    logger.info(
        "Wrote chunk %d: %d records, %.2f MB → %s",
        chunk_index,
        len(records),
        size_mb,
        dest_path,
    )


class StreamUploader:
    """
    Thread-safe local Parquet writer for extracted dataset records.

    Usage
    -----
    ::

        with StreamUploader() as up:
            for record in extractor.stream():
                up.add(record)
        # final flush happens on __exit__

    Parameters
    ----------
    output_dir : str, optional
        Directory to write Parquet chunks into.  Defaults to the ``output``
        folder one level above the repository root (set via ``OUTPUT_DIR``
        env var or the compiled-in ``_DEFAULT_OUTPUT_DIR``).
    chunk_size_mb : float
        Uncompressed JSON buffer size (MB) before auto-flush.
    split : str
        Sub-directory name used inside *output_dir* (e.g. ``"train"``).
    dry_run : bool
        If True, collect records but skip disk writes (for testing).
    repo_id : str, optional
        Accepted for API compatibility; not used for local storage.
    token : str, optional
        Accepted for API compatibility; not used for local storage.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        chunk_size_mb: float = _CHUNK_SIZE_MB,
        split: str = _HF_SPLIT,
        dry_run: bool = False,
        # kept for call-site compatibility; unused in local mode
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.output_dir = output_dir if output_dir is not None else _OUTPUT_DIR
        self.chunk_size_mb = chunk_size_mb
        self.split = split
        self.dry_run = dry_run
        # retained so existing callers that read these attrs don't break
        self.repo_id = repo_id or ""
        self.token = token or ""

        self._buffer: List[dict] = []
        self._buffer_bytes: int = 0
        self._chunk_index: int = 0
        self._total_records: int = 0
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add(self, record: dict) -> None:
        """Add a record; auto-flushes if buffer exceeds chunk_size_mb."""
        rb = _json_bytes(record)
        with self._lock:
            self._buffer.append(record)
            self._buffer_bytes += rb
            self._total_records += 1
            if self._buffer_bytes >= self.chunk_size_mb * _BYTES_PER_MB:
                self._flush_locked()

    def add_many(self, records) -> None:
        """Add an iterable of records."""
        for record in records:
            self.add(record)

    def flush(self) -> None:
        """Force-flush remaining buffered records."""
        with self._lock:
            self._flush_locked()

    @property
    def total_records(self) -> int:
        return self._total_records

    @property
    def chunk_count(self) -> int:
        return self._chunk_index

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _flush_locked(self) -> None:
        """Must be called with self._lock held."""
        if not self._buffer:
            return
        records = list(self._buffer)
        chunk_idx = self._chunk_index

        self._buffer = []
        self._buffer_bytes = 0
        self._chunk_index += 1

        if self.dry_run:
            logger.info(
                "[dry_run] Would write chunk %d: %d records", chunk_idx, len(records)
            )
            return

        try:
            _write_chunk_local(
                records=records,
                output_dir=self.output_dir,
                chunk_index=chunk_idx,
                split=self.split,
            )
        except Exception as exc:
            logger.error("Chunk %d write failed: %s", chunk_idx, exc)
            # Re-queue so data is not lost
            self._buffer = records + self._buffer
            self._buffer_bytes = sum(_json_bytes(r) for r in self._buffer)
            self._chunk_index -= 1
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        return False

    def __repr__(self) -> str:
        return (
            f"StreamUploader(output_dir={self.output_dir!r}, "
            f"records={self._total_records}, chunks={self._chunk_index}, "
            f"dry_run={self.dry_run})"
        )
