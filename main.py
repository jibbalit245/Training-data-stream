"""
main.py
Orchestration: launch 10 parallel extraction agents, stream results to HF,
and maintain a SQLite index of extracted documents.

Configuration is entirely environment-variable / config.yaml driven.
Secrets (HF_TOKEN, GITHUB_TOKEN, SE_API_KEY) are never hard-coded.

Stage 1 manifest mode
---------------------
Pass ``--stage1`` (or call ``run_stage1_pipeline``) to process the 73
documents listed in ``stage1_manifest.csv``.  Each row is dispatched to
the appropriate extractor based on its ``extractor_type`` column.
"""

from __future__ import annotations

import csv
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
from darwin_extractor import DarwinExtractor
from plato_extractor import PlatoExtractor

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
    output_dir: Optional[str] = None,
    num_agents: int = 10,
    dry_run: Optional[bool] = None,
    config_path: str = "config.yaml",
    enable_db: bool = True,
    # kept for call-site compatibility; unused in local mode
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Launch all extraction agents in parallel and write results to local disk.

    Parameters
    ----------
    output_dir : str, optional  Override OUTPUT_DIR (folder one level above repo).
    num_agents : int            Number of parallel agents (default 10).
    dry_run : bool, optional    Skip disk writes when True.
    config_path : str           Path to config.yaml.
    enable_db : bool            Create/update SQLite index.
    repo_id : str, optional     Accepted for compatibility; not used.
    token : str, optional       Accepted for compatibility; not used.

    Returns
    -------
    dict with pipeline summary.
    """
    cfg = _load_config(config_path)
    pip_cfg = cfg.get("pipeline", {})
    out_cfg = cfg.get("output", {})

    _output_dir = output_dir if output_dir is not None else (
        os.getenv("OUTPUT_DIR") or out_cfg.get("dir") or None
    )
    _dry = dry_run if dry_run is not None else (
        os.getenv("DRY_RUN", str(pip_cfg.get("dry_run", "false"))).lower()
        in ("1", "true", "yes")
    )
    dedup_threshold = float(os.getenv("DEDUP_THRESHOLD",
                                      str(pip_cfg.get("dedup_threshold", 0.92))))
    dedup_model = os.getenv("DEDUP_MODEL", pip_cfg.get("dedup_model", "all-MiniLM-L6-v2"))
    chunk_mb = float(os.getenv("HF_CHUNK_SIZE_MB",
                               str(out_cfg.get("chunk_size_mb", 50))))

    logger.info("Pipeline: %d agents | output_dir=%s | dry_run=%s",
                num_agents, _output_dir or "(default)", _dry)

    deduplicator = SemanticDeduplicator(threshold=dedup_threshold, model_name=dedup_model)
    uploader = StreamUploader(
        output_dir=_output_dir,
        chunk_size_mb=chunk_mb,
        dry_run=_dry,
    )
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
        "output_dir": uploader.output_dir,
        "dry_run": _dry,
        "db_summary": db_summary,
    }


# ---------------------------------------------------------------------------
# Stage 1 manifest support
# ---------------------------------------------------------------------------

# Map extractor_type column values to extractor classes
_STAGE1_EXTRACTOR_MAP: Dict[str, type] = {
    "DarwinXML": DarwinExtractor,
    "NewtonXML": DarwinExtractor,          # TEI XML correspondence
    "PlatoText": PlatoExtractor,
    "GalileoText": PlatoExtractor,         # dialogue-form text
    "AristotleText": PlatoExtractor,
    "HerschelText": PlatoExtractor,
    "FeynmanHTML": DialogueExtractor,
    "EuclidHTML": DialogueExtractor,
    "HTMLText": DialogueExtractor,
    "MahajanHTML": DialogueExtractor,
    "PDFText": PDFAcademicExtractor,
    "ExtractPDF": PDFAcademicExtractor,
    "MahajanPDF": PDFAcademicExtractor,
    "CyberneticsOpenAccess": PDFAcademicExtractor,
    "AutodeskDocs": DialogueExtractor,
    "OpenSCADDocs": DialogueExtractor,
    "BlenderDocs": DialogueExtractor,
    "GitHubJupyter": GitHubIssuesExtractor,
    "GitHubMarkdown": GitHubIssuesExtractor,
    # Copyrighted / book extracts require manual supply; skip gracefully
    "ExtractBook": None,
}


def load_stage1_manifest(manifest_path: str = "stage1_manifest.csv") -> List[Dict[str, str]]:
    """
    Load and return all rows from the Stage 1 source manifest CSV.

    Parameters
    ----------
    manifest_path : str
        Path to ``stage1_manifest.csv``.

    Returns
    -------
    list of dict
        Each dict has keys: doc_id, source, subject, tier, title, url,
        extractor_type, license, estimated_tokens, prerequisites,
        structural_prior.
    """
    rows: List[Dict[str, str]] = []
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(dict(row))
    return rows


def _build_stage1_agents(
    manifest_rows: List[Dict[str, str]],
    deduplicator: SemanticDeduplicator,
) -> List[Dict[str, Any]]:
    """
    Build extraction agent configurations from Stage 1 manifest rows.

    Groups rows by extractor_type and instantiates one agent per type,
    skipping types that are not yet implemented or are copyrighted.
    Passes manifest-assigned tiers as a URL→tier map so that each
    extractor can use ``manifest_tier`` in ``classify_tier()`` rather
    than falling back to the heuristic classifier.
    """
    # Group URLs and their manifest tiers by extractor type
    groups: Dict[str, List[str]] = {}
    tier_map: Dict[str, int] = {}  # url → manifest tier
    for row in manifest_rows:
        etype = row.get("extractor_type", "")
        url = row.get("url", "").strip()
        if not url or etype not in _STAGE1_EXTRACTOR_MAP:
            continue
        groups.setdefault(etype, []).append(url)
        try:
            tier_map[url] = int(row.get("tier", 1))
        except (ValueError, TypeError):
            tier_map[url] = 1

    agents = []
    for etype, urls in groups.items():
        cls = _STAGE1_EXTRACTOR_MAP.get(etype)
        if cls is None:
            logger.info("Stage 1: skipping extractor_type=%s (not yet implemented)", etype)
            continue

        agent_id = f"stage1_{etype.lower()}"
        # Build url→tier sub-map for this extractor's URLs
        extractor_tier_map = {u: tier_map[u] for u in urls if u in tier_map}

        # Instantiate with appropriate source keyword depending on class
        if issubclass(cls, (CorrespondenceExtractor, DarwinExtractor)):
            extractor = cls(
                xml_sources=urls,
                agent_id=agent_id,
                deduplicator=deduplicator,
                manifest_tier_map=extractor_tier_map,
            )
        elif issubclass(cls, (DialogueExtractor, PlatoExtractor)):
            extractor = cls(
                text_sources=urls,
                agent_id=agent_id,
                deduplicator=deduplicator,
                manifest_tier_map=extractor_tier_map,
            )
        elif issubclass(cls, PDFAcademicExtractor):
            extractor = cls(
                pdf_sources=urls,
                agent_id=agent_id,
                deduplicator=deduplicator,
                manifest_tier_map=extractor_tier_map,
            )
        elif issubclass(cls, GitHubIssuesExtractor):
            # GitHubIssuesExtractor takes repo slugs, not full URLs; skip
            logger.info("Stage 1: skipping GitHubIssuesExtractor (requires repo slugs)")
            continue
        else:
            logger.warning("Stage 1: unknown extractor class %s for type %s", cls, etype)
            continue

        agents.append({"name": f"stage1_{etype}", "extractor": extractor})

    return agents


def run_stage1_pipeline(
    manifest_path: str = "stage1_manifest.csv",
    output_dir: Optional[str] = None,
    dry_run: Optional[bool] = None,
    config_path: str = "config.yaml",
    enable_db: bool = True,
    # kept for call-site compatibility; unused in local mode
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the Stage 1 manifest-driven extraction pipeline.

    Reads ``stage1_manifest.csv``, groups documents by extractor type,
    and runs all extractors in parallel, writing results to local disk.

    Parameters
    ----------
    manifest_path : str
        Path to the Stage 1 source manifest CSV.
    output_dir : str, optional
        Override OUTPUT_DIR (folder one level above repo).
    dry_run : bool, optional
        Skip disk writes when True.
    config_path : str
        Path to config.yaml.
    enable_db : bool
        Create/update SQLite index.
    repo_id : str, optional
        Accepted for compatibility; not used.
    token : str, optional
        Accepted for compatibility; not used.

    Returns
    -------
    dict with pipeline summary including total_records and per-agent counts.
    """
    cfg = _load_config(config_path)
    out_cfg = cfg.get("output", {})

    _output_dir = output_dir if output_dir is not None else (
        os.getenv("OUTPUT_DIR") or out_cfg.get("dir") or None
    )
    _dry = dry_run if dry_run is not None else (
        os.getenv("DRY_RUN", str(cfg.get("pipeline", {}).get("dry_run", "false"))).lower()
        in ("1", "true", "yes")
    )
    dedup_threshold = float(os.getenv("DEDUP_THRESHOLD",
                                      str(cfg.get("pipeline", {}).get("dedup_threshold", 0.92))))
    dedup_model = os.getenv("DEDUP_MODEL",
                            cfg.get("pipeline", {}).get("dedup_model", "all-MiniLM-L6-v2"))
    chunk_mb = float(os.getenv("HF_CHUNK_SIZE_MB",
                               str(out_cfg.get("chunk_size_mb", 50))))

    manifest_rows = load_stage1_manifest(manifest_path)
    logger.info("Stage 1 pipeline: %d documents in manifest", len(manifest_rows))

    deduplicator = SemanticDeduplicator(threshold=dedup_threshold, model_name=dedup_model)
    uploader = StreamUploader(
        output_dir=_output_dir,
        chunk_size_mb=chunk_mb,
        dry_run=_dry,
    )
    indexer = DBIndexer() if enable_db else None

    agents = _build_stage1_agents(manifest_rows, deduplicator)
    num_agents = max(1, len(agents))

    logger.info("Stage 1: launching %d extraction agents | dry_run=%s", num_agents, _dry)

    results = []
    with uploader:
        with ThreadPoolExecutor(max_workers=num_agents, thread_name_prefix="stage1") as pool:
            futures = {
                pool.submit(run_agent, agent, uploader, indexer): agent["name"]
                for agent in agents
            }
            for future in as_completed(futures):
                aname = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.error("Stage 1 agent %s raised: %s", aname, exc)
                    results.append({"agent": aname, "count": 0, "errors": 1})

    total = sum(r.get("count", 0) for r in results)
    db_summary = indexer.summary() if indexer else {}
    logger.info("Stage 1 pipeline complete: %d total records", total)

    return {
        "total_records": total,
        "agents": results,
        "output_dir": uploader.output_dir,
        "dry_run": _dry,
        "db_summary": db_summary,
        "manifest_documents": len(manifest_rows),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if "--stage1" in sys.argv:
        manifest = "stage1_manifest.csv"
        for arg in sys.argv:
            if arg.startswith("--manifest="):
                manifest = arg.split("=", 1)[1]
        summary = run_stage1_pipeline(manifest_path=manifest)
        print(f"\nStage 1 done: {summary['total_records']} records from "
              f"{summary['manifest_documents']} manifest documents → {summary['output_dir']}")
    else:
        summary = run_pipeline()
        print(f"\nPipeline done: {summary['total_records']} records → {summary['output_dir']}")
    if summary.get("db_summary"):
        print(f"DB index: {summary['db_summary']}")
