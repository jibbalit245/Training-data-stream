"""
semantic_deduplicator.py
In-memory semantic deduplication for the extraction pipeline.

Strategy
--------
1. Fast exact SHA-256 fingerprint check (O(1)).
2. If not seen exactly, compute a sentence-transformer embedding and compare
   cosine similarity against all stored embeddings.
3. If similarity ≥ threshold → duplicate; else store and return False.

Memory model
------------
Embeddings are stored as a float32 numpy matrix grown incrementally.
At 384-dimensional all-MiniLM-L6-v2 embeddings each vector is 1.5 KB, so
1 million vectors ≈ 1.5 GB — well within the 28 GB per-agent budget.

Thread safety
-------------
A threading.Lock guards all state mutations so multiple extractor threads
can safely share one deduplicator instance.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("DEDUP_MODEL", "all-MiniLM-L6-v2")
DEFAULT_THRESHOLD = float(os.getenv("DEDUP_THRESHOLD", "0.92"))


class SemanticDeduplicator:
    """
    In-memory semantic deduplicator.

    Parameters
    ----------
    threshold : float
        Cosine similarity threshold above which a document is a duplicate.
    model_name : str
        Sentence-transformer model to use for embeddings.
    batch_encode_size : int
        How many texts to encode at once (for bulk seeding).
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        model_name: str = DEFAULT_MODEL,
        batch_encode_size: int = 64,
    ):
        self.threshold = threshold
        self.model_name = model_name
        self.batch_encode_size = batch_encode_size

        self._lock = threading.Lock()
        self._hashes: set[str] = set()
        self._embeddings: Optional[np.ndarray] = None  # shape (N, D)
        self._model = None
        self._model_failed = False

    # ------------------------------------------------------------------
    # Model accessor (lazy load)
    # ------------------------------------------------------------------
    def _get_model(self):
        if self._model is None and not self._model_failed:
            try:
                from sentence_transformers import SentenceTransformer  # lazy import
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                logger.warning(
                    "SemanticDeduplicator: cannot load model %r (%s). "
                    "Falling back to exact-hash-only deduplication.",
                    self.model_name, exc,
                )
                self._model_failed = True
        return self._model

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def is_duplicate(self, text: str) -> bool:
        """
        Return True if *text* is a near-duplicate of a previously seen doc.
        Thread-safe.  Falls back to exact-hash-only when the model is unavailable.
        """
        h = self._sha256(text)

        with self._lock:
            # 1. Exact hash
            if h in self._hashes:
                return True

            # 2. Semantic similarity (only when model is available)
            model = self._get_model()
            if model is not None:
                vec = model.encode([text], normalize_embeddings=True).astype(np.float32)
                if self._embeddings is not None and self._embeddings.shape[0] > 0:
                    sims = (self._embeddings @ vec.T).flatten()
                    if float(sims.max()) >= self.threshold:
                        return True
                # Not a duplicate — register embedding
                self._embeddings = (
                    vec if self._embeddings is None
                    else np.vstack([self._embeddings, vec])
                )

            # Not a duplicate — register hash
            self._hashes.add(h)
            return False

    def seed(self, texts: list[str]) -> None:
        """Pre-load a list of texts so they are treated as already seen."""
        if not texts:
            return
        model = self._get_model()
        for i in range(0, len(texts), self.batch_encode_size):
            batch = texts[i: i + self.batch_encode_size]
            hashes = [self._sha256(t) for t in batch]
            with self._lock:
                for h in hashes:
                    self._hashes.add(h)
                # Only encode embeddings when model is available
                if model is not None:
                    vecs = model.encode(batch, normalize_embeddings=True).astype(np.float32)
                    self._embeddings = (
                        vecs if self._embeddings is None
                        else np.vstack([self._embeddings, vecs])
                    )

    def reset(self) -> None:
        """Clear all state."""
        with self._lock:
            self._hashes.clear()
            self._embeddings = None
            self._model_failed = False

    @property
    def size(self) -> int:
        """Number of unique documents registered."""
        with self._lock:
            return len(self._hashes)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
