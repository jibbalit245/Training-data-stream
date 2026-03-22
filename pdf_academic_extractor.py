"""
pdf_academic_extractor.py
Extracts reasoning / argumentation passages from academic PDF files.

Uses pdfplumber (preferred) with pypdf as fallback.

Extraction targets
------------------
- Abstract, Introduction, Discussion, Conclusion sections
- Paragraphs containing explicit argumentative language
- Thought experiments and "What if" passages
- Boundary conditions and failure mode analyses
"""

from __future__ import annotations

import io
import logging
import re
from typing import Generator, List, Optional

from base_extractor import BaseExtractor, make_record, network_retry
from tier_classifier import classify_tier
from prior_tagger import tag_prior

logger = logging.getLogger(__name__)

# Sections that typically contain dense reasoning
_REASONING_SECTION_RE = re.compile(
    r"^\s*(abstract|introduction|discussion|conclusion|related work"
    r"|motivation|analysis|implications|theoretical framework"
    r"|background|methodology|proof|derivation)\b",
    re.IGNORECASE | re.MULTILINE,
)

# All-caps section headers (to detect end of reasoning section)
_SECTION_HEADER_RE = re.compile(r"^[A-Z][A-Z\s]{3,}$")

_ARGUMENT_RE = re.compile(
    r"\b(therefore|thus|hence|because|since|as a result|consequently"
    r"|we argue|we show|we propose|we demonstrate|we conclude|evidence suggests"
    r"|our approach|this indicates|in contrast|however|furthermore|moreover"
    r"|importantly|specifically|we claim|it follows|one can show)\b",
    re.IGNORECASE,
)

_THOUGHT_EXP_RE = re.compile(
    r"\b(what if|suppose|imagine|consider|let us assume|hypothetically"
    r"|thought experiment|for the sake of argument)\b",
    re.IGNORECASE,
)

_MIN_ARGUMENT_HITS = 1


def _has_reasoning(text: str) -> bool:
    return (
        len(_ARGUMENT_RE.findall(text)) >= _MIN_ARGUMENT_HITS
        or _THOUGHT_EXP_RE.search(text) is not None
    )


def _split_paragraphs(text: str) -> List[str]:
    paras = re.split(r"\n{2,}", text)
    return [p.replace("\n", " ").strip() for p in paras if p.strip()]


def _infer_domain(text: str) -> List[str]:
    domains = []
    lower = text.lower()
    mapping = {
        "physics": ["quantum", "particle", "field theory", "relativity", "entropy"],
        "math": ["theorem", "proof", "topology", "algebra", "calculus"],
        "biology": ["evolution", "gene", "protein", "cell", "organism"],
        "chemistry": ["molecule", "reaction", "bond", "chemical"],
        "computer_science": ["algorithm", "complexity", "neural network", "machine learning"],
        "philosophy": ["epistemology", "ontology", "metaphysics", "ethics"],
        "engineering": ["design", "system", "circuit", "mechanical"],
    }
    for domain, keywords in mapping.items():
        if any(kw in lower for kw in keywords):
            domains.append(domain)
    return domains or ["cross_domain"]


class PDFAcademicExtractor(BaseExtractor):
    """
    Extract reasoning passages from academic PDF files.

    Parameters
    ----------
    pdf_sources : list of str
        Local file paths or HTTP(S) URLs of PDF files.
    agent_id : str
    deduplicator : SemanticDeduplicator, optional
    """

    SOURCE_TYPE = "pdf_paper"

    def __init__(
        self,
        pdf_sources: Optional[List[str]] = None,
        agent_id: str = "agent_000",
        deduplicator=None,
        min_text_length: int = 100,
    ):
        super().__init__(
            agent_id=agent_id,
            deduplicator=deduplicator,
            min_text_length=min_text_length,
        )
        self.pdf_sources = pdf_sources or []

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    @network_retry
    def _load_bytes(self, source: str) -> bytes:
        if source.startswith(("http://", "https://")):
            import urllib.request
            with urllib.request.urlopen(source, timeout=60) as resp:  # noqa: S310
                return resp.read()
        with open(source, "rb") as fh:
            return fh.read()

    def _extract_text(self, pdf_bytes: bytes) -> str:
        # Try pdfplumber first (better layout handling)
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n\n".join(pages)
        except Exception:
            pass

        # Fallback: pypdf
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages)
        except Exception as exc:
            logger.error("Both PDF extractors failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Paragraph filtering
    # ------------------------------------------------------------------
    def _iter_reasoning_paragraphs(self, text: str) -> Generator[str, None, None]:
        in_reasoning = False
        for para in _split_paragraphs(text):
            if _REASONING_SECTION_RE.match(para):
                in_reasoning = True
                continue
            if _SECTION_HEADER_RE.match(para) and not _has_reasoning(para):
                in_reasoning = False
                continue
            if in_reasoning or _has_reasoning(para):
                yield para

    def _build_record(self, text: str, source: str) -> dict:
        domain = _infer_domain(text)
        doc_type = (
            "thought_experiment"
            if _THOUGHT_EXP_RE.search(text)
            else "proof" if re.search(r"\b(proof|theorem|lemma)\b", text, re.I)
            else "synthesis"
        )
        return make_record(
            raw_text=text,
            source_url=source,
            doc_type=doc_type,
            tier=classify_tier(text, doc_type=doc_type),
            license_="unknown",
            agent_id=self.agent_id,
        )

    # ------------------------------------------------------------------
    # BaseExtractor interface
    # ------------------------------------------------------------------
    def extract(
        self, pdf_sources: Optional[List[str]] = None
    ) -> Generator[dict, None, None]:
        sources = pdf_sources or self.pdf_sources
        for source in sources:
            try:
                pdf_bytes = self._load_bytes(source)
                text = self._extract_text(pdf_bytes)
                for para in self._iter_reasoning_paragraphs(text):
                    yield self._build_record(para, source)
            except Exception as exc:
                logger.error("PDFAcademicExtractor failed on %s: %s", source, exc)
