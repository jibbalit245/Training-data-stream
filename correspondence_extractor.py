"""
correspondence_extractor.py
Extracts reasoning passages and hypothesis co-development from
XML/TEI scientific correspondence.

Supported collections
---------------------
- Darwin Correspondence Project (TEI XML)
- Born-Einstein Letters (TEI or plain XML)
- Pauli Scientific Correspondence (TEI XML)
- Other scientific letter collections following TEI conventions

Extraction targets
------------------
- Reasoning chains and hypothesis proposals in letter bodies
- Challenge-response patterns across letter pairs
- "What if" speculation and thought experiments
- Passages where one correspondent updates/corrects another
"""

from __future__ import annotations

import logging
import os
import re
from typing import Generator, List, Optional
from xml.etree import ElementTree as ET

from base_extractor import BaseExtractor, make_record, network_retry
from tier_classifier import classify_tier
from prior_tagger import tag_prior

logger = logging.getLogger(__name__)

# TEI namespace
_TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

_REASONING_RE = re.compile(
    r"\b(therefore|hence|thus|because|conclude|hypothesis|evidence|argument"
    r"|theory|reason|natural selection|variation|believe|suppose|if|consequently"
    r"|speculate|what if|suppose|wonder|perhaps|might|could it be)\b",
    re.IGNORECASE,
)

# Minimum reasoning-keyword hits to accept a paragraph
_MIN_HITS = 2


def _collect_text(elem) -> str:
    """Recursively collect all text inside an XML element."""
    parts = []
    if elem.text:
        parts.append(elem.text.strip())
    for child in elem:
        parts.append(_collect_text(child))
        if child.tail:
            parts.append(child.tail.strip())
    return " ".join(p for p in parts if p)


def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _has_reasoning(text: str, min_hits: int = _MIN_HITS) -> bool:
    return len(_REASONING_RE.findall(text)) >= min_hits


def _extract_participants(root: ET.Element) -> List[str]:
    """Try to extract participant names from TEI header."""
    participants = []
    for ns in (_TEI_NS, {}):
        prefix = "tei:" if ns else ""
        for person in root.findall(f".//{prefix}persName", ns):
            name = (_collect_text(person) or "").strip()
            if name:
                participants.append(name.lower())
    return list(dict.fromkeys(participants))[:10]  # unique, max 10


def _infer_license(source: str) -> str:
    source_lower = source.lower()
    if "darwin" in source_lower:
        return "CC_BY"
    if "gutenberg" in source_lower:
        return "public_domain"
    return "unknown"


class CorrespondenceExtractor(BaseExtractor):
    """
    Extract reasoning passages from XML/TEI scientific correspondence.

    Parameters
    ----------
    xml_sources : list of str
        File paths or HTTP(S) URLs of XML/TEI files.
    agent_id : str
    deduplicator : SemanticDeduplicator, optional
    """

    SOURCE_TYPE = "correspondence"

    def __init__(
        self,
        xml_sources: Optional[List[str]] = None,
        agent_id: str = "agent_000",
        deduplicator=None,
        min_text_length: int = 100,
    ):
        super().__init__(
            agent_id=agent_id,
            deduplicator=deduplicator,
            min_text_length=min_text_length,
        )
        self.xml_sources = xml_sources or []

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    @network_retry
    def _load_xml(self, source: str) -> ET.Element:
        if source.startswith(("http://", "https://")):
            import urllib.request
            with urllib.request.urlopen(source, timeout=30) as resp:  # noqa: S310
                data = resp.read()
            return ET.fromstring(data)
        tree = ET.parse(source)
        return tree.getroot()

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------
    def _iter_paragraphs(self, root: ET.Element, source: str) -> Generator[dict, None, None]:
        participants = _extract_participants(root)
        license_ = _infer_license(source)
        seen_texts: set[str] = set()

        # Gather candidate elements: try TEI-namespaced divs first, fall back to bare tags
        tei_divs = root.findall(".//tei:div", _TEI_NS)
        if tei_divs:
            elems_to_scan = tei_divs
            tei_p = "tei:p"
            tei_ns: dict = _TEI_NS
        else:
            elems_to_scan = [
                e for e in root.iter()
                if _strip_ns(e.tag) in ("div", "letter", "body", "text")
            ]
            tei_p = ""
            tei_ns = {}

        for elem in elems_to_scan:
            # Iterate child <p> elements
            if tei_p:
                children = elem.findall(tei_p, tei_ns)
            else:
                children = [c for c in elem if _strip_ns(c.tag) == "p"]

            for p_elem in children:
                text = _collect_text(p_elem).strip()
                if text and _has_reasoning(text) and text not in seen_texts:
                    seen_texts.add(text)
                    yield self._build_record(text, source, participants, license_)

        # Also scan bare <p> elements at the top level (no containing div)
        top_p = (
            root.findall(".//tei:p", _TEI_NS) if tei_divs
            else [e for e in root.iter() if _strip_ns(e.tag) == "p"]
        )
        for p_elem in top_p:
            text = _collect_text(p_elem).strip()
            if text and _has_reasoning(text) and text not in seen_texts:
                seen_texts.add(text)
                yield self._build_record(text, source, participants, license_)

    def _build_record(
        self,
        text: str,
        source: str,
        participants: List[str],
        license_: str,
    ) -> dict:
        return make_record(
            raw_text=text,
            source_url=source,
            doc_type="correspondence",
            tier=classify_tier(text),
            structural_prior=tag_prior(text),
            domain=["cross_domain"],
            participants=participants,
            quality_score=min(1.0, len(_REASONING_RE.findall(text)) / 10),
            license_=license_,
            agent_id=self.agent_id,
        )

    # ------------------------------------------------------------------
    # BaseExtractor interface
    # ------------------------------------------------------------------
    def extract(
        self, xml_sources: Optional[List[str]] = None
    ) -> Generator[dict, None, None]:
        sources = xml_sources or self.xml_sources
        for source in sources:
            try:
                root = self._load_xml(source)
                yield from self._iter_paragraphs(root, source)
            except Exception as exc:
                logger.error("CorrespondenceExtractor failed on %s: %s", source, exc)
