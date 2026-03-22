"""
prior_tagger.py
Heuristic identification of the structural prior (0-9) for an extracted passage.

Structural prior codes
----------------------
0 = untagged             (no structural-prior signals detected)
1 = holographic_projection / duality
2 = tool_as_cognitive_prosthetic
3 = symmetry_breaking
4 = emergence / phase_transitions
5 = dimensional_analysis
6 = conservation_laws
7 = feedback_loops / control_theory
8 = information_geometry
9 = renormalization_group

Code 0 is intentionally distinct from code 1 so that the dream-state training
process can distinguish "this passage is genuinely about duality / holographic
projection" (code 1) from "we could not detect a structural prior" (code 0).
"""

from __future__ import annotations

import re
from typing import Optional

_PRIOR_PATTERNS = {
    1: re.compile(
        r"\b(duality|holograph|projection|AdS/CFT|bulk.boundary|mirror symmetry"
        r"|two sides|dual description|isomorphic|correspondence between"
        r"|one-to-one|bijection|mapping between)\b",
        re.IGNORECASE,
    ),
    2: re.compile(
        r"\b(tool|instrument|cognitive|prosthetic|scaffold|extended mind"
        r"|notation as|language as|diagram as|model as|using .{1,20} to think"
        r"|thinking with|medium of thought|computational thinking)\b",
        re.IGNORECASE,
    ),
    3: re.compile(
        r"\b(symmetry.break|spontaneous|phase transition|order parameter"
        r"|broken symmetry|Higgs|bifurcation|tipping point|asymmetry arises"
        r"|distinguish|differentiates)\b",
        re.IGNORECASE,
    ),
    4: re.compile(
        r"\b(emergence|emergent|phase transition|self.organiz|collective"
        r"|complex system|macro.behavior|micro.rules|critical point"
        r"|percolation|avalanche|scale.free)\b",
        re.IGNORECASE,
    ),
    5: re.compile(
        r"\b(dimension|unit|scaling|power law|dimensional analysis"
        r"|Buckingham|scale invariant|self.similar|fractal|ratio"
        r"|order of magnitude|Fermi estimate)\b",
        re.IGNORECASE,
    ),
    6: re.compile(
        r"\b(conservation|conserved quantity|invariant|Noether|symmetry implies"
        r"|flux|balance|steady.state|first law|entropy|energy balance"
        r"|mass balance|no.free.lunch)\b",
        re.IGNORECASE,
    ),
    7: re.compile(
        r"\b(feedback|control|loop|homeostasis|regulation|cybernetics"
        r"|servo|PID|stability|oscillat|attractor|equilibrium"
        r"|adaptive|negative feedback|positive feedback)\b",
        re.IGNORECASE,
    ),
    8: re.compile(
        r"\b(information|Fisher|entropy|KL.divergence|manifold|geometry"
        r"|statistical.manifold|metric|Riemannian|gradient|natural gradient"
        r"|exponential family|sufficient statistic)\b",
        re.IGNORECASE,
    ),
    9: re.compile(
        r"\b(renormali[sz]|coarse.grain|effective theory|relevant operator"
        r"|universality|fixed point|RG flow|Wilson|scale.dependent"
        r"|zoom out|integrate out|effective field)\b",
        re.IGNORECASE,
    ),
}


def tag_prior(text: str) -> int:
    """
    Return the structural prior code (0-9) with the highest keyword density.

    Ties broken by lower code (code 1 wins over 2, etc.).
    Returns 0 ("untagged") when no structural-prior signals are detected,
    so that downstream processes can distinguish genuinely classified passages
    from passages that could not be tagged.

    Returns
    -------
    int
        An integer in the range [0, 9]:
        0=untagged (no signals detected),
        1=holographic_projection/duality, 2=tool_as_cognitive_prosthetic,
        3=symmetry_breaking, 4=emergence/phase_transitions,
        5=dimensional_analysis, 6=conservation_laws,
        7=feedback_loops/control_theory, 8=information_geometry,
        9=renormalization_group.
    """
    scores = {code: len(pat.findall(text)) for code, pat in _PRIOR_PATTERNS.items()}
    max_score = max(scores.values())
    if max_score == 0:
        return 0  # untagged: no structural-prior signals detected
    for code in sorted(scores.keys()):
        if scores[code] == max_score:
            return code
    return 0
