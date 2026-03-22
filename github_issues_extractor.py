"""
github_issues_extractor.py
Extracts reasoning-heavy technical discussion threads from GitHub Issues.

Extraction targets
------------------
- Design decision debates with explicit rationale
- Bug investigation chains ("because", "therefore", "this means")
- Feature proposal discussions with challenge-response patterns
- Architecture debates and trade-off analyses
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

_GH_API = "https://api.github.com"

_DISCUSSION_LABELS = {
    "discussion", "design", "proposal", "rfc", "question",
    "needs discussion", "debate", "architecture", "decision",
    "help wanted", "feature request",
}

_REASONING_RE = re.compile(
    r"\b(because|therefore|however|alternatively|the reason|this means"
    r"|in order to|as a result|trade.off|pros and cons|approach|solution"
    r"|problem|issue|we should|I think|I believe|it seems|suggest|propose"
    r"|consider|conclude|evidence|argument)\b",
    re.IGNORECASE,
)


def _has_reasoning(text: str) -> bool:
    return len(_REASONING_RE.findall(text)) >= 2


def _build_auth_header(token: Optional[str]) -> dict:
    token = token or os.getenv("GITHUB_TOKEN", "")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "training-data-stream/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


class GitHubIssuesExtractor(BaseExtractor):
    """
    Extract reasoning-heavy issue threads from GitHub.

    Parameters
    ----------
    repos : list of str
        Repositories in "owner/repo" format.
    labels : list of str, optional
        Restrict to issues with these labels.
    max_issues_per_repo : int
    include_comments : bool
    github_token : str, optional
    agent_id : str
    deduplicator : SemanticDeduplicator, optional
    """

    SOURCE_TYPE = "github_issue"

    def __init__(
        self,
        repos: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        max_issues_per_repo: int = 100,
        include_comments: bool = True,
        github_token: Optional[str] = None,
        agent_id: str = "agent_000",
        deduplicator=None,
        min_text_length: int = 120,
    ):
        super().__init__(
            agent_id=agent_id,
            deduplicator=deduplicator,
            min_text_length=min_text_length,
        )
        self.repos = repos or []
        self.labels = labels
        self.max_issues_per_repo = max_issues_per_repo
        self.include_comments = include_comments
        self._headers = _build_auth_header(github_token)

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------
    @network_retry
    def _get_json(self, url: str, params: Optional[dict] = None) -> object:
        import urllib.request, urllib.parse
        if params:
            url = url + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers=self._headers)
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            return json.loads(resp.read())

    def _paginate_issues(self, owner: str, repo: str) -> Generator[dict, None, None]:
        page = 1
        fetched = 0
        while fetched < self.max_issues_per_repo:
            params: dict = {"state": "all", "per_page": 30, "page": page}
            if self.labels:
                params["labels"] = ",".join(self.labels)
            try:
                issues = self._get_json(
                    f"{_GH_API}/repos/{owner}/{repo}/issues", params=params
                )
            except Exception as exc:
                logger.error("GitHub API error %s/%s page %d: %s", owner, repo, page, exc)
                break

            if not isinstance(issues, list) or not issues:
                break

            for issue in issues:
                if isinstance(issue, dict) and "pull_request" not in issue:
                    yield issue
                    fetched += 1
                    if fetched >= self.max_issues_per_repo:
                        return
            page += 1
            time.sleep(1)

    def _get_comments(self, comments_url: str) -> List[dict]:
        try:
            result = self._get_json(comments_url)
            return result if isinstance(result, list) else []
        except Exception as exc:
            logger.warning("Failed to fetch comments %s: %s", comments_url, exc)
            return []

    def _is_discussion_issue(self, issue: dict) -> bool:
        labels = {lbl.get("name", "").lower() for lbl in issue.get("labels", [])}
        if labels & _DISCUSSION_LABELS:
            return True
        body = (issue.get("body") or "").strip()
        title = (issue.get("title") or "").strip()
        return _has_reasoning(body) or _has_reasoning(title)

    def _build_thread(self, issue: dict, comments: List[dict]) -> str:
        parts = [f"ISSUE: {issue.get('title', '')}\n\n{issue.get('body', '') or ''}"]
        for c in comments:
            body = (c.get("body") or "").strip()
            user = (c.get("user") or {}).get("login", "user")
            if body:
                parts.append(f"{user}: {body}")
        return "\n\n---\n\n".join(parts)

    def _build_record(self, thread: str, issue: dict, repo_full: str) -> dict:
        return make_record(
            raw_text=thread,
            source_url=issue.get("html_url", repo_full),
            doc_type="tool_demonstration",
            tier=classify_tier(thread, doc_type="tool_demonstration"),
            quality_score=min(1.0, len(_REASONING_RE.findall(thread)) / 12),
            license_="unknown",
            agent_id=self.agent_id,
            extra_metadata={
                "repo": repo_full,
                "issue_number": issue.get("number"),
                "issue_title": issue.get("title", ""),
                "issue_state": issue.get("state", ""),
            },
        )

    # ------------------------------------------------------------------
    # BaseExtractor interface
    # ------------------------------------------------------------------
    def extract(
        self, repos: Optional[List[str]] = None
    ) -> Generator[dict, None, None]:
        targets = repos or self.repos
        for repo_full in targets:
            parts = repo_full.split("/")
            if len(parts) != 2:
                logger.warning("Invalid repo format: %s", repo_full)
                continue
            owner, repo = parts
            for issue in self._paginate_issues(owner, repo):
                if not self._is_discussion_issue(issue):
                    continue
                comments: List[dict] = []
                if self.include_comments and issue.get("comments", 0) > 0:
                    comments = self._get_comments(issue.get("comments_url", ""))

                thread = self._build_thread(issue, comments)
                if not _has_reasoning(thread):
                    continue
                yield self._build_record(thread, issue, repo_full)
