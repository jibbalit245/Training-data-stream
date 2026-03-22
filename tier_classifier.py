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


def classify_tier(text: str) -> int:
    """
    Return the tier (1-5) best matching *text*.

    Scoring: count keyword hits per tier; pick the tier with the most hits.
    Ties are broken by preferring lower tier numbers (more fundamental).
    Default is 3 if no signals found.

    Returns
    -------
    int
        An integer in the range [1, 5]:
        1 = axioms/first principles, 2 = derived theorems,
        3 = applied models (default), 4 = computational/tool use,
        5 = meta-analysis/synthesis.
    """
    scores = {tier: len(pat.findall(text)) for tier, pat in _TIER_PATTERNS.items()}
    max_score = max(scores.values())
    if max_score == 0:
        return 3  # default: applied model
    # Among tied tiers, pick the lowest (most fundamental)
    for tier in sorted(scores.keys()):
        if scores[tier] == max_score:
            return tier
    return 3
