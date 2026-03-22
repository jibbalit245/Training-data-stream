"""
main.py
Orchestration: launch 10 parallel extraction agents, stream results to HF,
and maintain a SQLite index of extracted documents.

Configuration is entirely environment-variable / config.yaml driven.
Secrets (HF_TOKEN, GITHUB_TOKEN, SE_API_KEY) are never hard-coded.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

from semantic_deduplicator import SemanticDeduplicator
from stream_uploader import StreamUploader
from db_indexer import DBIndexer

from correspondence_extractor import CorrespondenceExtractor
from dialogue_extractor import DialogueExtractor
from pdf_academic_extractor import PDFAcademicExtractor
from arxiv_extractor import ArxivExtractor
from github_issues_extractor import GitHubIssuesExtractor
from stackexchange_extractor import StackExchangeExtractor

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def _load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()
    # Expand ${VAR} / ${VAR:-default} env references
    import re

    def _expand(match):
        var_expr = match.group(1)
        if ":-" in var_expr:
            var, default = var_expr.split(":-", 1)
            return os.getenv(var, default)
        return os.getenv(var_expr, "")

    expanded = re.sub(r"\$\{([^}]+)\}", _expand, raw)
    return yaml.safe_load(expanded) or {}


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
def _env_list(var: str, default: str = "") -> List[str]:
    raw = os.getenv(var, default)
    return [x.strip() for x in raw.split(",") if x.strip()]


def build_agents(
    cfg: dict,
    deduplicator: SemanticDeduplicator,
    num_agents: int = 10,
) -> List[Dict[str, Any]]:
    """
    Build extraction agent configurations from *cfg*.

    Returns up to *num_agents* agents spread across source types.
    """
    src = cfg.get("sources", {})

    # ---- Source lists -------------------------------------------------------
    correspondence_urls = _env_list("DARWIN_URLS") + _env_list("EINSTEIN_URLS") + _env_list("PAULI_URLS")
    correspondence_urls += [u for u in src.get("correspondence", {}).get("urls", []) if u]

    dialogue_urls = _env_list("PLATO_URLS") + _env_list("DIALOGUE_URLS")
    dialogue_urls += [u for u in src.get("dialogue", {}).get("urls", []) if u]

    pdf_urls = _env_list("PDF_URLS")
    pdf_urls += [u for u in src.get("pdf", {}).get("urls", []) if u]

    arxiv_cfg = src.get("arxiv", {})
    arxiv_query = os.getenv("ARXIV_QUERY", arxiv_cfg.get("query", "reasoning OR chain-of-thought"))
    arxiv_max = int(os.getenv("ARXIV_MAX_RESULTS", str(arxiv_cfg.get("max_results", 500))))
    arxiv_cats = arxiv_cfg.get("categories", ["cs.AI", "cs.CL", "math.LO", "physics.hist-ph"])

    gh_cfg = src.get("github", {})
    gh_repos = _env_list("GITHUB_REPOS") or gh_cfg.get("repos", [])
    gh_max = int(os.getenv("GITHUB_MAX_ISSUES", str(gh_cfg.get("max_issues_per_repo", 100))))

    se_cfg = src.get("stackexchange", {})
    se_sites = _env_list("SE_SITES") or se_cfg.get("sites", ["math", "physics", "philosophy", "cs"])
    se_min_score = int(os.getenv("SE_MIN_SCORE", str(se_cfg.get("min_score", 10))))
    se_max = int(os.getenv("SE_MAX_RESULTS", str(se_cfg.get("max_results_per_site", 200))))

    # ---- Split lists across two agents each --------------------------------
    def _half(lst, first=True):
        half = max(1, len(lst) // 2)
        return lst[:half] if first else lst[half:]

    agents = []

    # Correspondence agents (2)
    if src.get("correspondence", {}).get("enabled", True):
        agents += [
            {"name": "agent_001_correspondence",
             "extractor": CorrespondenceExtractor(
                 xml_sources=_half(correspondence_urls, True) or correspondence_urls,
                 agent_id="agent_001", deduplicator=deduplicator)},
            {"name": "agent_002_correspondence",
             "extractor": CorrespondenceExtractor(
                 xml_sources=_half(correspondence_urls, False) or correspondence_urls,
                 agent_id="agent_002", deduplicator=deduplicator)},
        ]

    # Dialogue agents (2)
    if src.get("dialogue", {}).get("enabled", True):
        agents += [
            {"name": "agent_003_dialogue",
             "extractor": DialogueExtractor(
                 text_sources=_half(dialogue_urls, True) or dialogue_urls,
                 agent_id="agent_003", deduplicator=deduplicator)},
            {"name": "agent_004_dialogue",
             "extractor": DialogueExtractor(
                 text_sources=_half(dialogue_urls, False) or dialogue_urls,
                 agent_id="agent_004", deduplicator=deduplicator)},
        ]

    # PDF agents (2)
    if src.get("pdf", {}).get("enabled", True):
        agents += [
            {"name": "agent_005_pdf",
             "extractor": PDFAcademicExtractor(
                 pdf_sources=_half(pdf_urls, True) or pdf_urls,
                 agent_id="agent_005", deduplicator=deduplicator)},
            {"name": "agent_006_pdf",
             "extractor": PDFAcademicExtractor(
                 pdf_sources=_half(pdf_urls, False) or pdf_urls,
                 agent_id="agent_006", deduplicator=deduplicator)},
        ]

    # arXiv agents (2)
    # agent_007 uses the primary query; agent_008 extends it with formal-reasoning
    # terms ("inference", "logic") to capture papers that use academic vocabulary
    # rather than the ML-centric "chain-of-thought" phrasing.
    # The agent_008 query extension can be overridden via ARXIV_QUERY_EXTENSION env var.
    arxiv_query_ext = os.getenv("ARXIV_QUERY_EXTENSION", "OR inference OR logic")
    if src.get("arxiv", {}).get("enabled", True):
        agents += [
            {"name": "agent_007_arxiv",
             "extractor": ArxivExtractor(
                 query=arxiv_query, categories=arxiv_cats,
                 max_results=arxiv_max // 2,
                 agent_id="agent_007", deduplicator=deduplicator)},
            {"name": "agent_008_arxiv",
             "extractor": ArxivExtractor(
                 query=f"{arxiv_query} {arxiv_query_ext}",
                 categories=arxiv_cats,
                 max_results=arxiv_max // 2,
                 agent_id="agent_008", deduplicator=deduplicator)},
        ]

    # GitHub agent (1)
    if src.get("github", {}).get("enabled", True):
        agents.append(
            {"name": "agent_009_github",
             "extractor": GitHubIssuesExtractor(
                 repos=gh_repos, max_issues_per_repo=gh_max,
                 agent_id="agent_009", deduplicator=deduplicator)}
        )

    # Stack Exchange agent (1)
    if src.get("stackexchange", {}).get("enabled", True):
        agents.append(
            {"name": "agent_010_stackexchange",
             "extractor": StackExchangeExtractor(
                 sites=se_sites, min_score=se_min_score,
                 max_results_per_site=se_max,
                 agent_id="agent_010", deduplicator=deduplicator)}
        )

    return agents[:num_agents]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def run_agent(
    agent: Dict[str, Any],
    uploader: StreamUploader,
    indexer: Optional[DBIndexer],
) -> Dict[str, Any]:
    name = agent["name"]
    extractor = agent["extractor"]
    count = 0
    errors = 0
    start = time.time()

    log_id = None
    if indexer:
        log_id = indexer.log_start(extractor.agent_id, name)

    logger.info("Agent %s starting", name)
    try:
        for record in extractor.stream():
            uploader.add(record)
            if indexer:
                try:
                    indexer.index_record(record)
                except Exception as idx_exc:
                    logger.debug("Index error: %s", idx_exc)
            count += 1
            if count % 200 == 0:
                logger.info("Agent %s: %d records so far", name, count)
    except Exception as exc:
        errors += 1
        logger.error("Agent %s error: %s", name, exc)
        if indexer and log_id:
            indexer.log_finish(log_id, "failed", count, str(exc))
        return {"agent": name, "count": count, "errors": errors,
                "elapsed": time.time() - start}

    elapsed = time.time() - start
    logger.info("Agent %s done: %d records, %.1fs", name, count, elapsed)
    if indexer and log_id:
        indexer.log_finish(log_id, "completed", count)
    return {"agent": name, "count": count, "errors": errors, "elapsed": elapsed}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_pipeline(
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    num_agents: int = 10,
    dry_run: Optional[bool] = None,
    config_path: str = "config.yaml",
    enable_db: bool = True,
) -> Dict[str, Any]:
    """
    Launch all extraction agents in parallel and stream results to HF.

    Parameters
    ----------
    repo_id : str, optional     Override HF_REPO_ID.
    token : str, optional       Override HF_TOKEN.
    num_agents : int            Number of parallel agents (default 10).
    dry_run : bool, optional    Skip HF upload when True.
    config_path : str           Path to config.yaml.
    enable_db : bool            Create/update SQLite index.

    Returns
    -------
    dict with pipeline summary.
    """
    cfg = _load_config(config_path)
    pip_cfg = cfg.get("pipeline", {})
    hf_cfg = cfg.get("huggingface", {})

    _repo = repo_id or hf_cfg.get("repo_id") or os.getenv("HF_REPO_ID", "")
    _token = token or hf_cfg.get("token") or os.getenv("HF_TOKEN", "")
    _dry = dry_run if dry_run is not None else (
        os.getenv("DRY_RUN", str(pip_cfg.get("dry_run", "false"))).lower()
        in ("1", "true", "yes")
    )
    dedup_threshold = float(os.getenv("DEDUP_THRESHOLD",
                                      str(pip_cfg.get("dedup_threshold", 0.92))))
    dedup_model = os.getenv("DEDUP_MODEL", pip_cfg.get("dedup_model", "all-MiniLM-L6-v2"))
    chunk_mb = float(os.getenv("HF_CHUNK_SIZE_MB",
                               str(hf_cfg.get("chunk_size_mb", 50))))

    logger.info("Pipeline: %d agents | repo=%s | dry_run=%s", num_agents, _repo, _dry)

    deduplicator = SemanticDeduplicator(threshold=dedup_threshold, model_name=dedup_model)
    uploader = StreamUploader(repo_id=_repo, token=_token, chunk_size_mb=chunk_mb, dry_run=_dry)
    indexer = DBIndexer() if enable_db else None

    agents = build_agents(cfg, deduplicator, num_agents)

    results = []
    with uploader:
        with ThreadPoolExecutor(max_workers=num_agents, thread_name_prefix="agent") as pool:
            futures = {
                pool.submit(run_agent, agent, uploader, indexer): agent["name"]
                for agent in agents
            }
            for future in as_completed(futures):
                aname = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.error("Agent %s raised: %s", aname, exc)
                    results.append({"agent": aname, "count": 0, "errors": 1})

    total = sum(r.get("count", 0) for r in results)
    db_summary = indexer.summary() if indexer else {}
    logger.info("Pipeline complete: %d total records", total)

    return {
        "total_records": total,
        "agents": results,
        "repo_id": _repo,
        "dry_run": _dry,
        "db_summary": db_summary,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    summary = run_pipeline()
    print(f"\nPipeline done: {summary['total_records']} records → {summary['repo_id']}")
    if summary.get("db_summary"):
        print(f"DB index: {summary['db_summary']}")
