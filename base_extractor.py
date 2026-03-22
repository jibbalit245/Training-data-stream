"""
base_extractor.py
Abstract base class for all extraction agents.

Responsibilities
----------------
- Define the canonical output record schema (with full metadata tagging)
- Provide retry-decorated I/O helpers
- Apply length filtering before yielding records
- Delegate deduplication to SemanticDeduplicator (passed in)
- Track agent ID for provenance

Output schema (per record)
--------------------------
{
    "doc_id": "<uuid>",
    "raw_content": "<original source text>",
    "text": "<|im_start|>system\\n...<|im_end|>\\n<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n<|im_end|>",
    "messages": [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": ""}
    ],
    "metadata": {
        "tier": 1,                       # 0-10 (0=unclassified, 1-5=knowledge, 6-10=application/scenario)
        "structural_prior": 3,           # 0-9  (0 = untagged, 1-9 = specific prior)
        "domain": ["physics", "math"],
        "doc_type": "correspondence",
        "participants": ["darwin", "hooker"],
        "quality_score": 0.87,           # 0.0-1.0
        "source_url": "...",
        "license": "public_domain",
        "extraction_timestamp": "...",
        "agent_id": "agent_001"
    }
}

Assistant-turn generation
--------------------------
The ``assistant`` turn in every record is intentionally left blank at
extraction time.  It is populated in a separate downstream step by
``observe_probe_generator.py``, which calls a capable model to produce
OBSERVE/PROBE completions.  The ``raw_content`` field preserves the
original source text so the generator can access it without parsing the
ChatML-formatted ``text`` field.

Token format
------------
Records use Qwen 2.5 ChatML tokens: ``<|im_start|>`` / ``<|im_end|>``.
Format::

    <|im_start|>system
    {system_prompt}<|im_end|>
    <|im_start|>user
    {user_content}<|im_end|>
    <|im_start|>assistant
    {assistant_content}<|im_end|>
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Generator, List, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from tier_prompts import get_tier_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry settings (overridable via env)
# ---------------------------------------------------------------------------
_MAX_ATTEMPTS = int(os.getenv("EXTRACTOR_MAX_RETRIES", "5"))
_MIN_WAIT = float(os.getenv("EXTRACTOR_RETRY_MIN_WAIT", "2"))
_MAX_WAIT = float(os.getenv("EXTRACTOR_RETRY_MAX_WAIT", "60"))

network_retry = retry(
    retry=retry_if_exception_type((IOError, OSError, ConnectionError, TimeoutError)),
    stop=stop_after_attempt(_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=_MIN_WAIT, max=_MAX_WAIT),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

# ---------------------------------------------------------------------------
# Valid metadata values
# ---------------------------------------------------------------------------
VALID_TIERS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
# 0 = unclassified
# 1-5 = knowledge tiers (assigned by classifier or manifest)
# 6-10 = application/scenario tiers (assigned manually, never by classifier)
# 0 = untagged (no structural-prior signals detected); 1-9 = specific priors
VALID_PRIORS = {0} | set(range(1, 10))
VALID_DOMAINS = {
    "physics", "math", "biology", "chemistry", "economics",
    "psychology", "engineering", "computer_science", "philosophy", "cross_domain",
}
VALID_DOC_TYPES = {
    "correspondence", "dialogue", "proof", "thought_experiment",
    "tool_demonstration", "boundary_condition_failure", "synthesis",
}
VALID_LICENSES = {
    "public_domain", "CC_BY", "CC_BY_SA", "copyrighted", "unknown",
}


# ---------------------------------------------------------------------------
# Record factory
# ---------------------------------------------------------------------------
def make_record(
    *,
    raw_text: str,
    source_url: str,
    doc_type: str,
    tier: int = 3,
    structural_prior: int = 1,
    domain: Optional[List[str]] = None,
    participants: Optional[List[str]] = None,
    quality_score: float = 0.5,
    license_: str = "unknown",
    agent_id: str = "agent_000",
    extra_metadata: Optional[dict] = None,
) -> dict:
    """
    Build a fully-typed output record.

    The ``text`` field uses the ChatML-like template required by the target
    model.  ``messages`` is the structured equivalent.
    """
    domain = domain or ["cross_domain"]
    participants = participants or []
    extra_metadata = extra_metadata or {}

    system_prompt = get_tier_prompt(tier)
    user_content = raw_text
    assistant_content = ""  # Generated in downstream observe_probe_generator.py

    formatted_text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_content}<|im_end|>"
    )

    return {
        "doc_id": str(uuid.uuid4()),
        "raw_content": raw_text,
        "text": formatted_text,
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "tier": tier,
            "structural_prior": structural_prior,
            "domain": domain,
            "doc_type": doc_type,
            "participants": participants,
            "quality_score": round(float(quality_score), 4),
            "source_url": source_url,
            "license": license_,
            "extraction_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "agent_id": agent_id,
            **extra_metadata,
        },
    }


# ---------------------------------------------------------------------------
# Base extractor
# ---------------------------------------------------------------------------
class BaseExtractor(ABC):
    """
    Abstract base class.  Subclasses implement :meth:`extract` which yields
    raw record dicts built with :func:`make_record`.

    The public :meth:`stream` method applies:
    1. Minimum text length filtering
    2. Semantic deduplication (via injected ``deduplicator``)
    """

    SOURCE_TYPE: str = "unknown"

    def __init__(
        self,
        agent_id: str = "agent_000",
        deduplicator=None,
        min_text_length: int = 80,
        manifest_tier_map: Optional[dict] = None,
    ):
        self.agent_id = agent_id
        self.deduplicator = deduplicator
        self.min_text_length = min_text_length
        self.manifest_tier_map: dict = manifest_tier_map or {}
        self._log = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public streaming API
    # ------------------------------------------------------------------
    def stream(self, *args, **kwargs) -> Generator[dict, None, None]:
        """Yield deduplicated, length-filtered records."""
        for record in self.extract(*args, **kwargs):
            raw_text = record.get("messages", [{}])[-1].get("content", "") \
                       or record.get("text", "")
            # Use the user content (target text) for length + dedup checks
            user_content = ""
            for msg in record.get("messages", []):
                if msg.get("role") == "user":
                    user_content = msg.get("content", "")
                    break
            check_text = user_content or raw_text

            if len(check_text) < self.min_text_length:
                continue
            if self.deduplicator is not None and self.deduplicator.is_duplicate(check_text):
                self._log.debug("Dedup skip: %s…", check_text[:60])
                continue
            yield record

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def extract(self, *args, **kwargs) -> Generator[dict, None, None]:
        """
        Yield records produced by :func:`make_record`.
        Implementations should apply ``@network_retry`` on I/O calls.
        """
