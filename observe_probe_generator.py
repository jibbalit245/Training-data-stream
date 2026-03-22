"""
observe_probe_generator.py
Second-pass processor: takes extracted JSONL records (with empty assistant
turns) and calls a capable model to generate OBSERVE/PROBE completions.

Responsibilities
----------------
- Read extracted JSONL records produced by the Stage 1 pipeline
- For each record, call the configured model to generate an OBSERVE/PROBE
  assistant turn
- Validate the 80/20 content ratio (raw_content vs observe_probe_text)
- Write completed records with filled assistant turns to an output JSONL file

Epistemic status tags used within OBSERVE/PROBE
------------------------------------------------
[GROUND]      — established, multiple sources confirm
[INFERENCE]   — follows from knowledge but unverified
[SPECULATION] — reasoning from limited information, needs testing
[BOUNDARY]    — don't know this, here's what I'd need to find out

Model options (in priority order)
----------------------------------
1. API call to Claude or GPT-4 (highest quality, costs money)
2. Local Qwen 2.5 72B-Instruct (if GPU memory is available)
3. Local Qwen 2.5 14B-Instruct (lower quality but free and fast)

Configuration is read from config.yaml (observe_probe section) and
environment variables.  The API key is never hard-coded.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Generation prompt
# ---------------------------------------------------------------------------
GENERATION_PROMPT = """You are generating training data for a reasoning model.
Given the following passage and its metadata, produce an OBSERVE section and
a PROBE section.

OBSERVE must:
- Identify what the author is DOING, not just what they are SAYING
- Note structural features: constraints, relationships, assumptions, methods
- Flag where the author holds multiple positions simultaneously
- Identify moments of insight, derivation, or paradigm shift
- Be 2-4 sentences. Dense. No filler. No summarizing.
- Use epistemic status tags where relevant: [GROUND] for established facts,
  [INFERENCE] for conclusions that follow from knowledge but are unverified,
  [SPECULATION] for reasoning from limited information that needs testing,
  [BOUNDARY] for the edges of what is known

PROBE must:
- Ask questions that FOLLOW from the observation
- Include at least one question connecting to another domain
- Include at least one question identifying where the framework might break
- Include at least one question suggesting a test or verification
- Be 2-4 questions. Each opens a line of inquiry.
- Use epistemic status tags where relevant

OBSERVE/PROBE must NOT:
- Summarize the content
- Praise the author
- Give answers
- Force connections that aren't structural
- Use hedge language

Passage metadata:
Tier: {tier}
Structural prior: {prior}
Domains: {domains}
Source type: {source_type}

Passage:
{raw_content}

Respond in EXACTLY this format:
[TIER:{tier}] [PRIOR:{prior}] [DOMAIN:{domains}] [SOURCE:{source_type}]

[OBSERVE] {{your observation}}

[PROBE] {{your questions}}"""

# ---------------------------------------------------------------------------
# Content ratio validation
# ---------------------------------------------------------------------------
_RATIO_MIN: float = 0.70
_RATIO_MAX: float = 0.85


def _simple_token_count(text: str) -> int:
    """Approximate token count by whitespace-splitting (no tokenizer required)."""
    return len(text.split())


def validate_content_ratio(
    raw_content: str,
    observe_probe_text: str,
    tokenizer=None,
) -> Tuple[bool, float]:
    """
    Validate that raw_content tokens represent 70–85 % of the combined token
    count (raw_content + observe_probe_text).

    Parameters
    ----------
    raw_content : str
        Original source text.
    observe_probe_text : str
        Generated OBSERVE/PROBE assistant turn.
    tokenizer : optional
        A tokenizer with an ``encode`` method.  Falls back to whitespace
        splitting when not provided.

    Returns
    -------
    (valid, ratio) where ``valid`` is True when ratio ∈ [0.70, 0.85].
    """
    if tokenizer is not None:
        content_tokens = len(tokenizer.encode(raw_content))
        meta_tokens = len(tokenizer.encode(observe_probe_text))
    else:
        content_tokens = _simple_token_count(raw_content)
        meta_tokens = _simple_token_count(observe_probe_text)

    total = content_tokens + meta_tokens
    if total == 0:
        return False, 0.0

    ratio = content_tokens / total
    valid = _RATIO_MIN <= ratio <= _RATIO_MAX
    return valid, ratio


# ---------------------------------------------------------------------------
# Model backends
# ---------------------------------------------------------------------------

def _call_api_model(
    prompt: str,
    model: str,
    api_key: str,
    max_tokens: int = 500,
    temperature: float = 0.3,
) -> str:
    """
    Call an OpenAI-compatible or Anthropic API.

    Supported ``model`` prefixes:
    - ``"api:claude"``  → Anthropic Claude (requires ``anthropic`` package)
    - ``"api:gpt"``     → OpenAI GPT (requires ``openai`` package)
      (also accepted as ``"api:gpt4"`` for convenience)
    """
    if model.startswith("api:claude"):
        import anthropic  # type: ignore
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()

    if model.startswith("api:gpt"):
        import openai  # type: ignore
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    raise ValueError(f"Unknown API model prefix: {model!r}")


def _call_local_model(
    prompt: str,
    model_path: str,
    max_tokens: int = 500,
    temperature: float = 0.3,
) -> str:
    """
    Call a local Qwen 2.5 Instruct model via the ``transformers`` pipeline.

    Requires ``transformers`` and ``torch`` (with sufficient GPU memory).
    """
    from transformers import pipeline as hf_pipeline  # type: ignore

    pipe = hf_pipeline(
        "text-generation",
        model=model_path,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        device_map="auto",
    )
    result = pipe(prompt)
    generated = result[0]["generated_text"]
    # Strip the prompt prefix that some pipelines echo back
    if generated.startswith(prompt):
        generated = generated[len(prompt):].strip()
    return generated


def generate_observe_probe(
    record: Dict[str, Any],
    model: str = "Qwen/Qwen2.5-14B-Instruct",
    api_key: str = "",
    max_tokens: int = 500,
    temperature: float = 0.3,
    tokenizer=None,
) -> Tuple[Optional[str], float]:
    """
    Generate an OBSERVE/PROBE assistant turn for *record*.

    Parameters
    ----------
    record : dict
        Extracted record with ``raw_content`` and ``metadata`` fields.
    model : str
        Model identifier.  Use ``"api:claude"`` or ``"api:gpt"`` (``"api:gpt4"``
        also accepted) for API calls; a HuggingFace model path for local inference.
    api_key : str
        API key (used only when ``model`` starts with ``"api:"``).
    max_tokens : int
        Maximum tokens for the generated output.
    temperature : float
        Sampling temperature (lower = more deterministic).
    tokenizer : optional
        Tokenizer used for content-ratio validation.

    Returns
    -------
    (observe_probe_text, ratio)
        ``observe_probe_text`` is None when generation fails or ratio is
        outside the accepted range.
    """
    raw_content = record.get("raw_content", "")
    meta = record.get("metadata", {})

    tier = meta.get("tier", 1)
    prior_code = meta.get("structural_prior", 0)
    domains = ", ".join(meta.get("domain", ["cross_domain"]))
    source_type = meta.get("doc_type", "unknown")

    from prior_tagger import PRIOR_NAMES
    prior_name = PRIOR_NAMES.get(prior_code, "none")

    prompt = GENERATION_PROMPT.format(
        tier=tier,
        prior=prior_name,
        domains=domains,
        source_type=source_type,
        raw_content=raw_content,
    )

    try:
        if model.startswith("api:"):
            text = _call_api_model(
                prompt, model, api_key, max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            text = _call_local_model(
                prompt, model, max_tokens=max_tokens,
                temperature=temperature,
            )
    except Exception as exc:
        logger.warning("Generation failed for doc_id=%s: %s", record.get("doc_id"), exc)
        return None, 0.0

    valid, ratio = validate_content_ratio(raw_content, text, tokenizer=tokenizer)
    if not valid:
        logger.debug(
            "Content ratio %.3f outside [%.2f, %.2f] for doc_id=%s — skipping",
            ratio, _RATIO_MIN, _RATIO_MAX, record.get("doc_id"),
        )
        return None, ratio

    return text, ratio


# ---------------------------------------------------------------------------
# Record completion
# ---------------------------------------------------------------------------

def complete_record(record: Dict[str, Any], observe_probe_text: str) -> Dict[str, Any]:
    """
    Return a copy of *record* with the assistant turn filled in.

    Updates both the ``text`` (ChatML) field and the ``messages`` list.
    """
    import copy
    completed = copy.deepcopy(record)

    # Update messages list
    for msg in completed.get("messages", []):
        if msg.get("role") == "assistant":
            msg["content"] = observe_probe_text
            break

    # Rebuild the ChatML text field
    msgs = completed.get("messages", [])
    system_content = next(
        (m["content"] for m in msgs if m["role"] == "system"), ""
    )
    user_content = next(
        (m["content"] for m in msgs if m["role"] == "user"), ""
    )
    completed["text"] = (
        f"<|im_start|>system\n{system_content}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{observe_probe_text}<|im_end|>"
    )

    return completed


# ---------------------------------------------------------------------------
# JSONL I/O helpers
# ---------------------------------------------------------------------------

def _iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def run_observe_probe_generator(
    input_path: str,
    output_path: str,
    model: str = "Qwen/Qwen2.5-14B-Instruct",
    api_key: str = "",
    max_tokens: int = 500,
    temperature: float = 0.3,
    tokenizer=None,
) -> Dict[str, Any]:
    """
    Process all records in *input_path* and write completed records to
    *output_path*.

    Parameters
    ----------
    input_path : str
        Path to a JSONL file of extracted records with empty assistant turns.
    output_path : str
        Path to write completed records.
    model : str
        Model to use for generation.
    api_key : str
        API key (for API-backed models).
    max_tokens : int
        Maximum tokens for generated OBSERVE/PROBE text.
    temperature : float
        Sampling temperature.
    tokenizer : optional
        Tokenizer for accurate content-ratio validation.

    Returns
    -------
    dict
        Summary with ``total``, ``completed``, ``skipped``, and
        ``ratio_violations`` counts.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    total = 0
    completed_count = 0
    skipped = 0
    ratio_violations = 0

    completed_records: List[Dict[str, Any]] = []

    for record in _iter_jsonl(input_path):
        total += 1
        doc_id = record.get("doc_id", f"unknown_{total}")

        observe_probe_text, ratio = generate_observe_probe(
            record,
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            tokenizer=tokenizer,
        )

        if observe_probe_text is None:
            if ratio != 0.0:
                ratio_violations += 1
            else:
                skipped += 1
            logger.info("Skipped doc_id=%s (ratio=%.3f)", doc_id, ratio)
            continue

        completed = complete_record(record, observe_probe_text)
        completed_records.append(completed)
        completed_count += 1
        logger.info(
            "Completed doc_id=%s (ratio=%.3f) [%d/%d]",
            doc_id, ratio, completed_count, total,
        )

    _write_jsonl(output_path, completed_records)

    summary = {
        "total": total,
        "completed": completed_count,
        "skipped": skipped,
        "ratio_violations": ratio_violations,
        "output_path": output_path,
    }
    logger.info(
        "observe_probe_generator done: %d/%d completed, %d ratio violations, %d skipped",
        completed_count, total, ratio_violations, skipped,
    )
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate OBSERVE/PROBE assistant turns for extracted records."
    )
    parser.add_argument("input", help="Input JSONL file (extracted records)")
    parser.add_argument("output", help="Output JSONL file (completed records)")
    parser.add_argument(
        "--model",
        default=os.getenv("OBSERVE_PROBE_MODEL", "Qwen/Qwen2.5-14B-Instruct"),
        help="Model to use (api:claude, api:gpt4, or HF model path)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OBSERVE_PROBE_API_KEY", ""),
        help="API key for api:* models",
    )
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    summary = run_observe_probe_generator(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(
        f"\nobserve_probe_generator complete:\n"
        f"  total={summary['total']}\n"
        f"  completed={summary['completed']}\n"
        f"  ratio_violations={summary['ratio_violations']}\n"
        f"  skipped={summary['skipped']}\n"
        f"  output={summary['output_path']}"
    )
