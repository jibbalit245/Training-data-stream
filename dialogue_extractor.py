"""
dialogue_extractor.py
Extracts Socratic / dialectical reasoning passages from plain-text works.

Supported sources
-----------------
- Plato's Complete Dialogues (any translation, plain text)
- Galileo's Two New Sciences
- Berkeley's Three Dialogues
- Hume's Dialogues Concerning Natural Religion
- Lakatos's "Proofs and Refutations"

Extraction targets
------------------
- Challenge-response patterns (A proposes, B objects, A refines)
- Hypothesis co-development exchanges
- Thought experiments introduced with "What if / Suppose / Imagine"
- Passages where a definition is tested and revised
"""

from __future__ import annotations

import logging
import re
from typing import Generator, List, Optional, Tuple

from base_extractor import BaseExtractor, make_record, network_retry
from tier_classifier import classify_tier
from prior_tagger import tag_prior

logger = logging.getLogger(__name__)

# Pattern matching SPEAKER. rest-of-line  (e.g. "SOCRATES. Indeed,")
_SPEAKER_RE = re.compile(
    r"^([A-Z][A-Z\s\-\.]{1,30}[A-Z])\.\s+(.*)",
    re.MULTILINE,
)

# Thought-experiment triggers
_THOUGHT_EXP_RE = re.compile(
    r"\b(what if|suppose|imagine|let us assume|consider a case|hypothetically"
    r"|thought experiment|if we were|were it the case|if it should happen"
    r"|pretend|for the sake of argument)\b",
    re.IGNORECASE,
)

# Challenge / rebuttal markers
_CHALLENGE_RE = re.compile(
    r"\b(but|however|yet|on the contrary|I disagree|objection|not so fast"
    r"|wait|hold on|you say|you claim|it seems to me|is it not the case"
    r"|what of|how then|I grant|I concede|surely not)\b",
    re.IGNORECASE,
)

_PHILOSOPHY_RE = re.compile(
    r"\b(virtue|justice|knowledge|soul|truth|wisdom|good|argument|definition"
    r"|form|idea|reason|examine|consider|necessary|impossible|certainly"
    r"|agree|contradict|therefore|indeed|beauty|piety|courage|temperance)\b",
    re.IGNORECASE,
)

_MIN_TURNS = 3   # minimum speaker turns per output chunk
_CHUNK_TURNS = 6  # target turns per chunk


def _text_has_dialectic(text: str) -> bool:
    return (
        len(_PHILOSOPHY_RE.findall(text)) >= 2
        or _THOUGHT_EXP_RE.search(text) is not None
        or _CHALLENGE_RE.search(text) is not None
    )


def _infer_participants(turns: List[Tuple[str, str]]) -> List[str]:
    seen = dict.fromkeys(spk.lower() for spk, _ in turns)
    return list(seen.keys())[:10]


class DialogueExtractor(BaseExtractor):
    """
    Extract Socratic / dialectical exchanges from plain-text dialogue works.

    Parameters
    ----------
    text_sources : list of str
        File paths or HTTP(S) URLs of plain-text files.
    chunk_turns : int
        Number of speaker turns per output record.
    agent_id : str
    deduplicator : SemanticDeduplicator, optional
    """

    SOURCE_TYPE = "dialogue"

    def __init__(
        self,
        text_sources: Optional[List[str]] = None,
        chunk_turns: int = _CHUNK_TURNS,
        agent_id: str = "agent_000",
        deduplicator=None,
        min_text_length: int = 120,
    ):
        super().__init__(
            agent_id=agent_id,
            deduplicator=deduplicator,
            min_text_length=min_text_length,
        )
        self.text_sources = text_sources or []
        self.chunk_turns = chunk_turns

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    @network_retry
    def _load_text(self, source: str) -> str:
        if source.startswith(("http://", "https://")):
            import urllib.request
            with urllib.request.urlopen(source, timeout=30) as resp:  # noqa: S310
                return resp.read().decode("utf-8", errors="replace")
        with open(source, encoding="utf-8", errors="replace") as fh:
            return fh.read()

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _parse_turns(self, text: str) -> List[Tuple[str, str]]:
        """Parse text into (speaker, utterance) tuples."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        turns: List[Tuple[str, str]] = []
        current_speaker: Optional[str] = None
        current_lines: List[str] = []

        for line in text.splitlines():
            m = _SPEAKER_RE.match(line)
            if m:
                if current_speaker is not None:
                    utt = " ".join(current_lines).strip()
                    if utt:
                        turns.append((current_speaker, utt))
                current_speaker = m.group(1).strip()
                current_lines = [m.group(2).strip()]
            elif current_speaker is not None:
                stripped = line.strip()
                if stripped:
                    current_lines.append(stripped)

        if current_speaker and current_lines:
            utt = " ".join(current_lines).strip()
            if utt:
                turns.append((current_speaker, utt))
        return turns

    def _iter_dialogue_chunks(
        self, text: str, source: str
    ) -> Generator[dict, None, None]:
        turns = self._parse_turns(text)

        if len(turns) < _MIN_TURNS:
            # Fallback: paragraph-based extraction
            for para in re.split(r"\n{2,}", text):
                para = para.strip()
                if para and _text_has_dialectic(para):
                    yield self._build_record(para, source, [])
            return

        step = max(1, self.chunk_turns // 2)
        for i in range(0, len(turns) - self.chunk_turns + 1, step):
            window = turns[i: i + self.chunk_turns]
            chunk = "\n".join(f"{spk}: {utt}" for spk, utt in window)
            if _text_has_dialectic(chunk):
                participants = _infer_participants(window)
                doc_type = (
                    "thought_experiment"
                    if _THOUGHT_EXP_RE.search(chunk)
                    else "dialogue"
                )
                yield self._build_record(chunk, source, participants, doc_type)

    def _build_record(
        self,
        text: str,
        source: str,
        participants: List[str],
        doc_type: str = "dialogue",
    ) -> dict:
        return make_record(
            raw_text=text,
            source_url=source,
            doc_type=doc_type,
            tier=classify_tier(text),
            structural_prior=tag_prior(text),
            domain=["philosophy"],
            participants=participants,
            quality_score=min(1.0, (
                len(_PHILOSOPHY_RE.findall(text))
                + len(_CHALLENGE_RE.findall(text))
            ) / 15),
            license_="public_domain",
            agent_id=self.agent_id,
        )

    # ------------------------------------------------------------------
    # BaseExtractor interface
    # ------------------------------------------------------------------
    def extract(
        self, text_sources: Optional[List[str]] = None
    ) -> Generator[dict, None, None]:
        sources = text_sources or self.text_sources
        for source in sources:
            try:
                text = self._load_text(source)
                yield from self._iter_dialogue_chunks(text, source)
            except Exception as exc:
                logger.error("DialogueExtractor failed on %s: %s", source, exc)
