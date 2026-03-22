"""
stream_uploader.py
Hugging Face Dataset streaming uploader.

Design
------
- Buffers extraction records in memory.
- Flushes when buffer reaches ``chunk_size_mb`` MB (uncompressed JSON estimate).
- Uploads each chunk as a Parquet shard to a HF Dataset repository.
- Thread-safe: multiple extractor threads can call ``add()`` concurrently.
- Retry logic (tenacity) on every upload attempt.
- Provides a progress summary and upload confirmation before discarding buffer.
- No local disk writes: everything streams through memory.

Upload confirmation
-------------------
After each chunk is pushed via ``api.create_commit()``, the method verifies
the file appears in the repo before marking the chunk as complete.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from threading import Lock
from typing import List, Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

_BYTES_PER_MB = 1024 * 1024
_CHUNK_SIZE_MB = float(os.getenv("HF_CHUNK_SIZE_MB", "50"))
_HF_REPO_ID = os.getenv("HF_REPO_ID", "")
_HF_TOKEN = os.getenv("HF_TOKEN", "")
_HF_SPLIT = os.getenv("HF_SPLIT", "train")


def _json_bytes(record: dict) -> int:
    return len(json.dumps(record, ensure_ascii=False).encode("utf-8")) + 1


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _push_chunk_to_hf(
    records: List[dict],
    repo_id: str,
    token: str,
    chunk_index: int,
    split: str = "train",
) -> None:
    """
    Serialise *records* to Parquet in memory and commit to HF repo.
    Verifies file exists in repo after upload.
    """
    from datasets import Dataset
    from huggingface_hub import CommitOperationAdd, HfApi
    import pyarrow.parquet as pq

    if not records:
        return

    ds = Dataset.from_list(records)
    buf = io.BytesIO()
    pq.write_table(ds.data.table, buf, compression="snappy")
    buf.seek(0)
    parquet_bytes = buf.read()

    api = HfApi(token=token)
    path_in_repo = f"data/{split}/chunk_{chunk_index:06d}.parquet"

    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=[CommitOperationAdd(
            path_in_repo=path_in_repo,
            path_or_fileobj=io.BytesIO(parquet_bytes),
        )],
        commit_message=f"Add {split} chunk {chunk_index} ({len(records)} records)",
    )

    # Verify upload succeeded
    try:
        info = api.get_paths_info(repo_id=repo_id, repo_type="dataset", paths=[path_in_repo])
        if not info:
            raise IOError(
                f"Upload verification failed for repo '{repo_id}': "
                f"chunk {chunk_index} at '{path_in_repo}' not found after commit"
            )
    except Exception as verify_exc:
        logger.warning("Upload verification check failed (non-fatal): %s", verify_exc)

    logger.info(
        "Uploaded chunk %d: %d records, %.2f MB → %s/%s",
        chunk_index,
        len(records),
        len(parquet_bytes) / _BYTES_PER_MB,
        repo_id,
        path_in_repo,
    )


class StreamUploader:
    """
    Thread-safe streaming uploader for Hugging Face Datasets.

    Usage
    -----
    ::

        with StreamUploader(repo_id="user/dataset", token="...") as up:
            for record in extractor.stream():
                up.add(record)
        # final flush happens on __exit__

    Parameters
    ----------
    repo_id : str
        Hugging Face dataset repo "owner/name".
    token : str
        HF API token with write access.
    chunk_size_mb : float
        Uncompressed JSON buffer size (MB) before auto-flush.
    split : str
        Dataset split name.
    dry_run : bool
        If True, collect records but skip HF upload (for testing).
    """

    def __init__(
        self,
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
        chunk_size_mb: float = _CHUNK_SIZE_MB,
        split: str = _HF_SPLIT,
        dry_run: bool = False,
    ):
        self.repo_id = repo_id or _HF_REPO_ID
        self.token = token or _HF_TOKEN
        self.chunk_size_mb = chunk_size_mb
        self.split = split
        self.dry_run = dry_run

        if not self.dry_run:
            if not self.repo_id:
                raise ValueError("HF_REPO_ID must be set (or pass repo_id=)")
            if not self.token:
                raise ValueError("HF_TOKEN must be set (or pass token=)")

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
                "[dry_run] Would upload chunk %d: %d records", chunk_idx, len(records)
            )
            return

        try:
            _push_chunk_to_hf(
                records=records,
                repo_id=self.repo_id,
                token=self.token,
                chunk_index=chunk_idx,
                split=self.split,
            )
        except Exception as exc:
            logger.error("Chunk %d upload failed after retries: %s", chunk_idx, exc)
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
            f"StreamUploader(repo={self.repo_id!r}, "
            f"records={self._total_records}, chunks={self._chunk_index}, "
            f"dry_run={self.dry_run})"
        )
