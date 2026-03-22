"""
darwin_extractor.py
Darwin Correspondence Project extractor.

Extracts scientific reasoning passages from Darwin Correspondence Project
XML/TEI letters. Identifies observations, hypotheses, and reasoning chains.

Wraps the existing CorrespondenceExtractor with Stage 1 manifest awareness.
"""

from __future__ import annotations

import logging
import re
from typing import Generator, List, Optional

from correspondence_extractor import CorrespondenceExtractor
from base_extractor import make_record, network_retry
from tier_classifier import classify_tier
from prior_tagger import tag_prior

logger = logging.getLogger(__name__)

# Darwin-specific reasoning markers in addition to those in CorrespondenceExtractor
_DARWIN_REASONING_RE = re.compile(
    r"\b(observed|observation|hypothesis|speculate|conjecture"
    r"|therefore|thus|hence|consequently|evidence|fact"
    r"|however|but|although|yet|question|query|wonder"
    r"|species|variety|variation|structure|form|adaptation"
    r"|natural selection|divergence|descent|modification)\b",
    re.IGNORECASE,
)

_MIN_DARWIN_HITS = 2


class DarwinExtractor(CorrespondenceExtractor):
    """
    Extractor specialised for Darwin Correspondence Project XML/TEI letters.

    Applies Darwin-specific reasoning keyword scoring on top of the base
    CorrespondenceExtractor XML parsing.

    Parameters
    ----------
    xml_sources : list of str
        File paths or HTTP(S) URLs of Darwin Correspondence Project XML files.
    agent_id : str
    deduplicator : SemanticDeduplicator, optional
    min_text_length : int
        Minimum character count for a passage to be kept (default 200).
    """

    SOURCE_TYPE = "darwin_correspondence"

    def __init__(
        self,
        xml_sources: Optional[List[str]] = None,
        agent_id: str = "agent_000",
        deduplicator=None,
        min_text_length: int = 200,
    ):
        super().__init__(
            xml_sources=xml_sources,
            agent_id=agent_id,
            deduplicator=deduplicator,
            min_text_length=min_text_length,
        )

    # ------------------------------------------------------------------
    # Override record builder to apply Darwin-specific scoring
    # ------------------------------------------------------------------
    def _build_record(
        self,
        text: str,
        source: str,
        participants: List[str],
        license_: str,
    ) -> dict:
        hits = len(_DARWIN_REASONING_RE.findall(text))
        return make_record(
            raw_text=text,
            source_url=source,
            doc_type="correspondence",
            tier=classify_tier(text, doc_type="correspondence"),
            participants=participants,
            quality_score=min(1.0, hits / 12),
            license_=license_,
            agent_id=self.agent_id,
        )

    @staticmethod
    def _has_darwin_reasoning(text: str) -> bool:
        """Return True if the passage has enough Darwin reasoning markers."""
        return len(_DARWIN_REASONING_RE.findall(text)) >= _MIN_DARWIN_HITS

    def extract(
        self, xml_sources: Optional[List[str]] = None
    ) -> Generator[dict, None, None]:
        """
        Extract reasoning passages from Darwin XML sources.

        Applies Darwin-specific reasoning filter after the base XML parsing.
        """
        for record in super().extract(xml_sources):
            # Further filter: ensure the passage has Darwin-specific vocabulary
            user_content = ""
            for msg in record.get("messages", []):
                if msg.get("role") == "user":
                    user_content = msg.get("content", "")
                    break
            if self._has_darwin_reasoning(user_content):
                yield record
