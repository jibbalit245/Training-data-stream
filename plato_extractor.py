"""
plato_extractor.py
Plato Dialogue extractor.

Extracts Socratic exchanges from Plato's dialogues and other philosophical
texts in dialogue form. Wraps DialogueExtractor with Plato/Socratic-specific
vocabulary scoring.

Supported sources
-----------------
- Plato's dialogues (Gutenberg plain-text or HTML)
- Hume's Dialogues Concerning Natural Religion
- Berkeley's Three Dialogues
- Lakatos's Proofs and Refutations
"""

from __future__ import annotations

import logging
import re
from typing import Generator, List, Optional, Tuple

from dialogue_extractor import DialogueExtractor
from base_extractor import make_record
from tier_classifier import classify_tier
from prior_tagger import tag_prior

logger = logging.getLogger(__name__)

# Socratic / dialectic vocabulary
_SOCRATIC_RE = re.compile(
    r"\b(virtue|justice|knowledge|soul|truth|wisdom|good|argument|definition"
    r"|form|idea|reason|examine|consider|necessary|impossible|certainly"
    r"|agree|contradict|therefore|indeed|beauty|piety|courage|temperance"
    r"|dialectic|logos|eidos|aporia|elenchus|maieutics|hypothesis"
    r"|what is|can you tell|do you think|is it not|how then|must we not)\b",
    re.IGNORECASE,
)

_MIN_SOCRATIC_HITS = 2


class PlatoExtractor(DialogueExtractor):
    """
    Extractor specialised for Platonic dialogues and Socratic method texts.

    Applies Socratic-specific vocabulary scoring on top of the base
    DialogueExtractor turn-parsing logic.

    Parameters
    ----------
    text_sources : list of str
        File paths or HTTP(S) URLs of plain-text or HTML dialogue files.
    chunk_turns : int
        Number of speaker turns per output record (default 6).
    agent_id : str
    deduplicator : SemanticDeduplicator, optional
    min_text_length : int
        Minimum character count for a chunk to be kept (default 120).
    """

    SOURCE_TYPE = "plato_dialogues"

    def __init__(
        self,
        text_sources: Optional[List[str]] = None,
        chunk_turns: int = 6,
        agent_id: str = "agent_000",
        deduplicator=None,
        min_text_length: int = 120,
    ):
        super().__init__(
            text_sources=text_sources,
            chunk_turns=chunk_turns,
            agent_id=agent_id,
            deduplicator=deduplicator,
            min_text_length=min_text_length,
        )

    # ------------------------------------------------------------------
    # Override record builder to apply Plato-specific scoring
    # ------------------------------------------------------------------
    def _build_record(
        self,
        text: str,
        participants: List[str],
        doc_type: str = "dialogue",
    ) -> dict:
        hits = len(_SOCRATIC_RE.findall(text))
        return make_record(
            raw_text=text,
            source_url="",          # caller fills source_url via super()
            doc_type=doc_type,
            tier=classify_tier(text, doc_type=doc_type),
            structural_prior=tag_prior(text),
            domain=["philosophy"],
            participants=participants,
            quality_score=min(1.0, hits / 15),
            license_="public_domain",
            agent_id=self.agent_id,
        )

    # Override to pass source through to _build_record
    def _build_record_with_source(
        self,
        text: str,
        source: str,
        participants: List[str],
        doc_type: str = "dialogue",
    ) -> dict:
        hits = len(_SOCRATIC_RE.findall(text))
        return make_record(
            raw_text=text,
            source_url=source,
            doc_type=doc_type,
            tier=classify_tier(text, doc_type=doc_type),
            structural_prior=tag_prior(text),
            domain=["philosophy"],
            participants=participants,
            quality_score=min(1.0, hits / 15),
            license_="public_domain",
            agent_id=self.agent_id,
        )

    @staticmethod
    def _has_socratic_content(text: str) -> bool:
        """Return True if the chunk has enough Socratic vocabulary."""
        return len(_SOCRATIC_RE.findall(text)) >= _MIN_SOCRATIC_HITS

    def extract(
        self, text_sources: Optional[List[str]] = None
    ) -> Generator[dict, None, None]:
        """
        Extract Socratic dialogue chunks from plain-text sources.

        Applies Plato-specific vocabulary filter on top of base extraction.
        """
        sources = text_sources or self.text_sources
        for source in sources:
            try:
                text = self._load_text(source)
                for record in self._iter_dialogue_chunks(text, source):
                    user_content = ""
                    for msg in record.get("messages", []):
                        if msg.get("role") == "user":
                            user_content = msg.get("content", "")
                            break
                    # Keep only records with Socratic vocabulary content
                    if self._has_socratic_content(user_content):
                        yield record
            except Exception as exc:
                logger.error("PlatoExtractor failed on %s: %s", source, exc)
