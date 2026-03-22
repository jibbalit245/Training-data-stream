"""
arxiv_extractor.py
Extracts reasoning-rich passages from arXiv papers via the arXiv API
(Atom feed) with optional full-text PDF processing.

Rate limiting: arXiv requests ≤ 1 per 3 seconds (enforced here).
"""

from __future__ import annotations

import logging
import re
import time
from typing import Generator, List, Optional
from xml.etree import ElementTree as ET

from base_extractor import BaseExtractor, make_record, network_retry
from tier_classifier import classify_tier
from prior_tagger import tag_prior

logger = logging.getLogger(__name__)

_API_URL = "https://export.arxiv.org/api/query"
_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"

_REASONING_RE = re.compile(
    r"\b(reasoning|argumentation|inference|logic|proof|theorem|hypothesis"
    r"|evidence|causal|explanation|chain-of-thought|deduction|induction"
    r"|abduction|thought experiment|analogy|cross.domain)\b",
    re.IGNORECASE,
)

_DEFAULT_CATEGORIES = [
    "cs.AI", "cs.CL", "cs.LG", "math.LO",
    "physics.hist-ph", "quant-ph", "hep-th",
]


def _abstract_is_relevant(abstract: str) -> bool:
    return bool(_REASONING_RE.search(abstract))


def _infer_domain(text: str, categories: List[str]) -> List[str]:
    cat_map = {
        "cs": "computer_science",
        "math": "math",
        "physics": "physics",
        "q-bio": "biology",
        "econ": "economics",
        "stat": "math",
    }
    domains = []
    for cat in categories:
        prefix = cat.split(".")[0]
        if prefix in cat_map:
            domains.append(cat_map[prefix])
    return list(dict.fromkeys(domains)) or ["cross_domain"]


class ArxivExtractor(BaseExtractor):
    """
    Stream reasoning-focused paper abstracts (and optionally full text)
    from the arXiv API.

    Parameters
    ----------
    query : str
        Free-text arXiv query.
    categories : list of str
        arXiv categories to include.
    max_results : int
        Total results to fetch (paginated).
    fetch_pdf : bool
        If True, download and extract full paper text.
    agent_id : str
    deduplicator : SemanticDeduplicator, optional
    """

    SOURCE_TYPE = "arxiv"

    def __init__(
        self,
        query: str = "reasoning OR chain-of-thought OR argumentation",
        categories: Optional[List[str]] = None,
        max_results: int = 100,
        fetch_pdf: bool = False,
        agent_id: str = "agent_000",
        deduplicator=None,
        min_text_length: int = 100,
    ):
        super().__init__(
            agent_id=agent_id,
            deduplicator=deduplicator,
            min_text_length=min_text_length,
        )
        self.query = query
        self.categories = categories or _DEFAULT_CATEGORIES
        self.max_results = max_results
        self.fetch_pdf = fetch_pdf

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------
    @network_retry
    def _fetch_page(self, start: int, page_size: int) -> ET.Element:
        import urllib.request, urllib.parse
        cat_filter = " OR ".join(f"cat:{c}" for c in self.categories)
        full_query = f"({self.query}) AND ({cat_filter})"
        params = urllib.parse.urlencode({
            "search_query": full_query,
            "start": start,
            "max_results": page_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        })
        url = f"{_API_URL}?{params}"
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
            data = resp.read()
        return ET.fromstring(data)

    def _parse_entries(self, root: ET.Element) -> List[dict]:
        ns = {"atom": _ATOM_NS, "arxiv": _ARXIV_NS}
        entries = []
        for entry in root.findall("atom:entry", ns):
            arxiv_id = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            abstract = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            abstract = abstract.replace("\n", " ")
            published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()

            categories = [
                lnk.get("term", "") for lnk in entry.findall("atom:category", ns)
            ]

            pdf_url = None
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")
                    if pdf_url and not pdf_url.endswith(".pdf"):
                        pdf_url += ".pdf"
                    break

            if not abstract or not _abstract_is_relevant(abstract):
                continue

            entries.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "published": published,
                "categories": categories,
                "pdf_url": pdf_url,
            })
        return entries

    @network_retry
    def _fetch_pdf_text(self, pdf_url: str) -> str:
        from pdf_academic_extractor import PDFAcademicExtractor
        ext = PDFAcademicExtractor(agent_id=self.agent_id)
        pdf_bytes = ext._load_bytes(pdf_url)
        return ext._extract_text(pdf_bytes)

    def _build_record(self, entry: dict, text: str) -> dict:
        domain = _infer_domain(text, entry.get("categories", []))
        return make_record(
            raw_text=text,
            source_url=entry.get("arxiv_id", ""),
            doc_type="synthesis",
            tier=classify_tier(text, doc_type="synthesis"),
            license_="CC_BY",
            agent_id=self.agent_id,
            extra_metadata={
                "title": entry["title"],
                "published": entry["published"],
                "pdf_url": entry.get("pdf_url", ""),
                "arxiv_categories": entry.get("categories", []),
            },
        )

    # ------------------------------------------------------------------
    # BaseExtractor interface
    # ------------------------------------------------------------------
    def extract(self) -> Generator[dict, None, None]:
        page_size = min(100, self.max_results)
        fetched = 0

        while fetched < self.max_results:
            try:
                root = self._fetch_page(start=fetched, page_size=page_size)
            except Exception as exc:
                logger.error("arXiv API failed at offset %d: %s", fetched, exc)
                break

            entries = self._parse_entries(root)
            if not entries:
                break

            for entry in entries:
                text = f"{entry['title']}\n\n{entry['abstract']}"

                if self.fetch_pdf and entry.get("pdf_url"):
                    try:
                        pdf_text = self._fetch_pdf_text(entry["pdf_url"])
                        if pdf_text and len(pdf_text) > len(text):
                            text = pdf_text
                    except Exception as exc:
                        logger.warning("PDF fetch failed for %s: %s", entry["arxiv_id"], exc)

                yield self._build_record(entry, text)
                fetched += 1
                if fetched >= self.max_results:
                    break

            time.sleep(3)  # arXiv rate limit
