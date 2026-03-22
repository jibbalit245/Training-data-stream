"""
stackexchange_extractor.py
Extracts high-quality Q&A threads from Stack Exchange sites via the
Stack Exchange API v2.3.

Extraction targets
------------------
- High-voted questions with accepted answers (reasoning-rich explanations)
- Cross-domain analogies in answers
- Derivations and proof-style answers on Math/Physics/CS
- "Why does X work?" explanatory threads
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Generator, List, Optional

from base_extractor import BaseExtractor, make_record, network_retry
from tier_classifier import classify_tier
from prior_tagger import tag_prior

logger = logging.getLogger(__name__)

_SE_API = "https://api.stackexchange.com/2.3"
_PAGESIZE = 100

_REASONING_RE = re.compile(
    r"\b(because|therefore|thus|hence|proof|derive|follows from|assume"
    r"|suppose|consider|it can be shown|we have|therefore|consequently"
    r"|in particular|note that|observe that|intuitively|this is because"
    r"|the key insight|the reason)\b",
    re.IGNORECASE,
)

_SITE_DOMAIN_MAP = {
    "math": "math",
    "physics": "physics",
    "philosophy": "philosophy",
    "cs": "computer_science",
    "stats": "math",
    "chemistry": "chemistry",
    "biology": "biology",
    "economics": "economics",
    "psychology": "psychology",
    "crossvalidated": "math",
    "datascience": "computer_science",
    "ai": "computer_science",
}

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s{2,}")


def _strip_html(html: str) -> str:
    text = _HTML_TAG_RE.sub(" ", html)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _has_reasoning(text: str) -> bool:
    return len(_REASONING_RE.findall(text)) >= 2


class StackExchangeExtractor(BaseExtractor):
    """
    Extract high-quality reasoning Q&A from Stack Exchange sites.

    Parameters
    ----------
    sites : list of str
        SE site names e.g. ["math", "physics", "philosophy"].
    min_score : int
        Minimum question score to include.
    max_results_per_site : int
        Maximum questions per site.
    api_key : str, optional
        SE API key (increases rate limit). Use SE_API_KEY env var.
    agent_id : str
    deduplicator : SemanticDeduplicator, optional
    """

    SOURCE_TYPE = "stackexchange"

    def __init__(
        self,
        sites: Optional[List[str]] = None,
        min_score: int = 10,
        max_results_per_site: int = 200,
        api_key: Optional[str] = None,
        agent_id: str = "agent_000",
        deduplicator=None,
        min_text_length: int = 150,
        manifest_tier_map: Optional[dict] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            deduplicator=deduplicator,
            min_text_length=min_text_length,
            manifest_tier_map=manifest_tier_map,
        )
        self.sites = sites or ["math", "physics", "philosophy", "cs"]
        self.min_score = min_score
        self.max_results_per_site = max_results_per_site
        self._api_key = api_key or os.getenv("SE_API_KEY", "")

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------
    def _base_params(self) -> dict:
        p: dict = {"filter": "withbody", "pagesize": _PAGESIZE}
        if self._api_key:
            p["key"] = self._api_key
        return p

    @network_retry
    def _get_json(self, url: str, params: Optional[dict] = None) -> dict:
        import urllib.request, urllib.parse
        all_params = {**self._base_params(), **(params or {})}
        full_url = url + "?" + urllib.parse.urlencode(all_params)
        with urllib.request.urlopen(full_url, timeout=30) as resp:  # noqa: S310
            raw = resp.read()
        # SE API responses may be gzip-compressed
        try:
            import gzip
            raw = gzip.decompress(raw)
        except Exception:
            pass
        return json.loads(raw)

    def _iter_questions(self, site: str) -> Generator[dict, None, None]:
        page = 1
        fetched = 0
        while fetched < self.max_results_per_site:
            try:
                data = self._get_json(
                    f"{_SE_API}/questions",
                    params={
                        "site": site,
                        "sort": "votes",
                        "order": "desc",
                        "min": self.min_score,
                        "page": page,
                    },
                )
            except Exception as exc:
                logger.error("SE API error site=%s page=%d: %s", site, page, exc)
                break

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                yield item
                fetched += 1
                if fetched >= self.max_results_per_site:
                    return

            if not data.get("has_more", False):
                break

            page += 1
            # SE API throttle: backoff if requested
            backoff = data.get("backoff", 0)
            time.sleep(max(0.5, float(backoff)))

    @network_retry
    def _get_answers(self, question_id: int, site: str) -> List[dict]:
        data = self._get_json(
            f"{_SE_API}/questions/{question_id}/answers",
            params={"site": site, "sort": "votes", "order": "desc"},
        )
        return data.get("items", [])

    def _build_thread(self, question: dict, answers: List[dict]) -> str:
        q_body = _strip_html(question.get("body", "") or "")
        q_title = question.get("title", "")
        parts = [f"QUESTION: {q_title}\n\n{q_body}"]
        for ans in answers[:3]:  # top-3 answers
            a_body = _strip_html(ans.get("body", "") or "")
            accepted = "✓ " if ans.get("is_accepted", False) else ""
            parts.append(f"{accepted}ANSWER (score {ans.get('score', 0)}):\n{a_body}")
        return "\n\n---\n\n".join(parts)

    def _build_record(self, thread: str, question: dict, site: str) -> dict:
        domain = _SITE_DOMAIN_MAP.get(site, "cross_domain")
        question_url = question.get("link", f"https://{site}.stackexchange.com")
        return make_record(
            raw_text=thread,
            source_url=question_url,
            doc_type="synthesis",
            tier=classify_tier(thread, doc_type="synthesis",
                               manifest_tier=self.manifest_tier_map.get(question_url)),
            quality_score=min(1.0, (question.get("score", 0) or 0) / 100),
            license_="CC_BY_SA",
            agent_id=self.agent_id,
            extra_metadata={
                "site": site,
                "question_id": question.get("question_id"),
                "question_score": question.get("score", 0),
                "view_count": question.get("view_count", 0),
                "tags": question.get("tags", []),
            },
        )

    # ------------------------------------------------------------------
    # BaseExtractor interface
    # ------------------------------------------------------------------
    def extract(self) -> Generator[dict, None, None]:
        for site in self.sites:
            for question in self._iter_questions(site):
                q_body = _strip_html(question.get("body", "") or "")
                if not _has_reasoning(q_body):
                    continue

                answers: List[dict] = []
                q_id = question.get("question_id")
                if q_id and question.get("answer_count", 0) > 0:
                    try:
                        answers = self._get_answers(q_id, site)
                    except Exception as exc:
                        logger.warning("Failed to fetch answers for q%s: %s", q_id, exc)

                thread = self._build_thread(question, answers)
                if not _has_reasoning(thread):
                    continue
                yield self._build_record(thread, question, site)
