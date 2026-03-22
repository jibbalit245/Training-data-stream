"""
tier_classifier.py
Heuristic-based classifier that assigns a tier (1-5) to extracted text.

Tier definitions
----------------
1 = Axioms / first principles  (foundational definitions, postulates)
2 = Derived theorems           (proofs, formal derivations)
3 = Applied models             (worked examples, calculations)
4 = Computational / tool use   (code, algorithms, numerical experiments)
5 = Meta-analysis / synthesis  (cross-domain bridges, reviews, commentary)

Scoring algorithm
-----------------
A pure keyword-count approach misclassifies primary reasoning documents (e.g.
scientific correspondence or Socratic dialogue) that mention "observation" or
"experiment" while presenting foundational logical arguments.  The classifier
therefore uses three additional signals beyond raw keyword hits:

1. **Reasoning chain density** – logical connectors (therefore, hence, because,
   it follows, etc.) per word indicate structured argument rather than applied
   description.  Their density is added as a bonus to tier-1 and tier-2 scores.

2. **Document-type hint** – when the caller knows the source document type
   (e.g. "correspondence", "dialogue", "proof") it can pass ``doc_type``.
   Foundational source types receive a 1.5× multiplier on tiers 1 and 2.

3. **Tier-3 margin guard** – tier 3 must exceed the best foundational score
   (tier 1 or 2) by a meaningful margin to "win".  This prevents a passage
   with a few incidental "observation" mentions from overriding a tier-2 signal
   that is reinforced by reasoning-chain density.
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Keyword signals per tier
# ---------------------------------------------------------------------------
_TIER_PATTERNS = {
    1: re.compile(
        r"\b(axiom|postulate|definition|first principle|fundamental|by definition"
        r"|let us define|we define|given that|assume that|suppose that"
        r"|premise|a priori)\b",
        re.IGNORECASE,
    ),
    2: re.compile(
        r"\b(theorem|lemma|corollary|proof|QED|q\.e\.d\.|therefore|hence"
        r"|it follows that|we can show|derives from|implies that|necessarily"
        r"|by induction|by contradiction|suppose for contradiction)\b",
        re.IGNORECASE,
    ),
    3: re.compile(
        r"\b(model|apply|application|example|calculate|computation|simulate"
        r"|experiment|measure|observation|empirically|in practice"
        r"|for instance|let us consider|we compute|numerical)\b",
        re.IGNORECASE,
    ),
    4: re.compile(
        r"\b(algorithm|code|function|variable|implementation|program"
        r"|software|library|API|debug|compile|runtime|data structure"
        r"|complexity|Big-O|benchmark)\b",
        re.IGNORECASE,
    ),
    5: re.compile(
        r"\b(synthesis|analogy|cross-domain|bridge|unify|connection between"
        r"|perspective|review|reflect|meta|paradigm|framework|insight"
        r"|broader implication|in summary|this suggests|what this means)\b",
        re.IGNORECASE,
    ),
}

# ---------------------------------------------------------------------------
# Reasoning-chain density signals (boost tiers 1 and 2)
# ---------------------------------------------------------------------------
# These logical connectors signal that the text is structured as an argument
# (derivation, proof sketch, or reasoned hypothesis) rather than an empirical
# description or worked application.
_LOGICAL_CONNECTOR_RE = re.compile(
    r"\b(therefore|hence|thus|it follows|we conclude|consequently"
    r"|because|since|given that|implies|derives from|necessarily"
    r"|we can show|as a result|from this|from which)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
# Chain-density bonus for tiers 1 and 2:
#   bonus = min(_CHAIN_BONUS_CAP, (connectors / words) * _CHAIN_DENSITY_SCALE)
_CHAIN_BONUS_CAP: float = 3.0      # maximum bonus added to tier-1 and tier-2 scores
_CHAIN_DENSITY_SCALE: float = 50.0  # 1 connector per 50 words → +1.0 bonus

# Foundational doc-type multiplier applied to tier-1 and tier-2 raw+bonus scores
_FOUNDATIONAL_MULTIPLIER: float = 1.5

# Tier-3 margin guard thresholds:
#   tier-3 must reach best_foundational + max(_MIN_TIER3_MARGIN,
#                                             best_foundational * _TIER3_MARGIN_RATIO)
_MIN_TIER3_MARGIN: float = 2.0
_TIER3_MARGIN_RATIO: float = 0.5

# Floating-point tolerance for tie-breaking in the final argmax scan
_SCORE_TOLERANCE: float = 1e-9

# ---------------------------------------------------------------------------
# Document-type hint sets
# ---------------------------------------------------------------------------
# Source types whose content is predominantly foundational reasoning.
_FOUNDATIONAL_DOC_TYPES = frozenset({
    "correspondence", "dialogue", "proof", "thought_experiment",
})


def classify_tier(
    text: str,
    doc_type: Optional[str] = None,
    manifest_tier: Optional[int] = None,
) -> int:
    """
    Return the tier (1-5) best matching *text*.

    Parameters
    ----------
    text : str
        The passage to classify.
    doc_type : str, optional
        The document type from the extraction manifest
        (e.g. "correspondence", "dialogue", "proof", "synthesis").
        When provided for foundational source types, tiers 1 and 2 receive
        a 1.5× multiplier so that primary reasoning documents are not
        pushed into tier 3 by incidental application vocabulary.
    manifest_tier : int, optional
        When provided and within VALID_TIERS, this value is returned
        directly without running the heuristic classifier.  Manifest
        assignments always take precedence over heuristic classification.

    Returns
    -------
    int
        An integer in the range [1, 5]:
        1 = axioms/first principles, 2 = derived theorems,
        3 = applied models (default), 4 = computational/tool use,
        5 = meta-analysis/synthesis.
    """
    from base_extractor import VALID_TIERS
    if manifest_tier is not None and manifest_tier in VALID_TIERS:
        return manifest_tier
    # 1. Raw keyword scores (float so subsequent arithmetic stays consistent)
    scores: dict[int, float] = {
        tier: float(len(pat.findall(text)))
        for tier, pat in _TIER_PATTERNS.items()
    }

    # 2. Reasoning-chain density bonus for tiers 1 and 2.
    #    Logical connectors per word signal structured argument.  Their density
    #    is scaled to a bonus capped at _CHAIN_BONUS_CAP per tier.
    words = text.split()
    if words:
        connector_count = len(_LOGICAL_CONNECTOR_RE.findall(text))
        chain_density = connector_count / len(words)
        chain_bonus = min(_CHAIN_BONUS_CAP, chain_density * _CHAIN_DENSITY_SCALE)
        scores[1] += chain_bonus
        scores[2] += chain_bonus

    # 3. Document-type hint: foundational source types get a _FOUNDATIONAL_MULTIPLIER
    #    on tiers 1 and 2, reflecting that correspondence and dialogue are
    #    primary sources of logical reasoning, not applied description.
    if doc_type in _FOUNDATIONAL_DOC_TYPES:
        scores[1] *= _FOUNDATIONAL_MULTIPLIER
        scores[2] *= _FOUNDATIONAL_MULTIPLIER

    # 4. Tier-3 margin guard.
    #    Tier 3 must exceed the best foundational score by a meaningful margin
    #    (the larger of _MIN_TIER3_MARGIN or _TIER3_MARGIN_RATIO × that score)
    #    to win the classification.  Without this guard, a passage with several
    #    incidental "observation" / "experiment" mentions overrides a tier-2
    #    signal reinforced by reasoning-chain density — misclassifying primary
    #    derivation documents.
    best_foundational = max(scores[1], scores[2])
    if scores[3] > 0 and best_foundational > 0:
        required_margin = max(_MIN_TIER3_MARGIN, best_foundational * _TIER3_MARGIN_RATIO)
        if scores[3] < best_foundational + required_margin:
            scores[3] = best_foundational - 0.01

    max_score = max(scores.values())
    if max_score <= 0:
        return 3  # default: applied model
    # Among tied tiers, pick the lowest (most fundamental)
    for tier in sorted(scores.keys()):
        if scores[tier] >= max_score - _SCORE_TOLERANCE:
            return tier
    return 3
