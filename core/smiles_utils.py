# -*- coding: utf-8 -*-
"""
SMILES 工具函数：
- 多组分/多片段 SMILES 分列
- RDKit canonical 化
- 配方键生成
- [新增] SMILES 清洗与智能修复
"""

from __future__ import annotations

# ============================================
# 重要：必须在导入 RDKit 之前导入线程配置！
# ============================================
from . import thread_config

import re
from functools import lru_cache
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem.SaltRemover import SaltRemover

    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False
    Chem = None
    SaltRemover = None

# -----------------------------------------------------------------------------
# Optional deps: SELFIES / BigSMILES
# -----------------------------------------------------------------------------
try:
    import selfies as sf  # type: ignore
    SELFIES_AVAILABLE = True
except Exception:
    sf = None
    SELFIES_AVAILABLE = False

try:
    import bigsmiles  # type: ignore
    BIGSMILES_AVAILABLE = True
except Exception:
    bigsmiles = None
    BIGSMILES_AVAILABLE = False

_SELFIES_PATTERN = re.compile(r"^\s*(\[[^\[\]]+\])+\s*$")
_SELFIES_TOKEN_RE = re.compile(r"\[[^\[\]]+\]")
_BIGSMILES_HINT = re.compile(r"[{}]")
_BIGSMILES_BOND_TOKEN_RE = re.compile(r"\[\s*(?:<|>|\*|\$)(?:[^\]]*)\]")
_BIGSMILES_EMPTY_TOKEN_RE = re.compile(r"\[\s*\]")
_BIGSMILES_INLINE_META_RE = re.compile(r"\|[^|]*\|")
_PSEUDO_REPEAT_BRACKET_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9()=#@+\-]{3,})\]")
_NAKED_CONNECTOR_RE = re.compile(r"[<>]")
_ISOLATED_METAL_RING_RE = re.compile(
    r"^\s*(\[(?:Co|Cu|Ni|Fe|Zn|Mn|Cr|Al|Ti|V)(?:[+-]\d+)?\])(?:\d|%\d{2})+\s*$"
)


def _looks_bigsmiles_like(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return False
    if _BIGSMILES_HINT.search(s):
        return True
    if _BIGSMILES_BOND_TOKEN_RE.search(s):
        return True
    if _NAKED_CONNECTOR_RE.search(s):
        return True
    return False

def detect_chem_string_format(text: str) -> str:
    """Detect whether the input looks like SMILES / SELFIES / BigSMILES.

    Returns: one of {'smiles','selfies','bigsmiles','empty'}
    """
    if text is None:
        return "empty"
    s = str(text).strip()
    if not s or s.lower() in {"nan", "none", "<na>", "na", "-block-"}:
        return "empty"
    # Accept common explicit prefixes
    s2 = re.sub(r"^(SMILES|SELFIES|BIGSMILES)\s*[:?]\s*", "", s, flags=re.I).strip()
    explicit_selfies = bool(re.match(r"^\s*SELFIES\s*[:?]\s*", s, flags=re.I))
    if _SELFIES_PATTERN.match(s2):
        tokens = _SELFIES_TOKEN_RE.findall(s2)
        # Avoid misclassifying bracketed SMILES like [Li+], [Cl-], [Co+2] as SELFIES.
        if explicit_selfies or len(tokens) >= 2:
            return "selfies"
    if _looks_bigsmiles_like(s2):
        # BigSMILES typically contains curly braces { } or polymer bonding descriptors.
        return "bigsmiles"
    return "smiles"


def _repair_phosphine_oxide_h(smiles: str) -> Optional[str]:
    s = clean_smiles_raw_string(smiles)
    if not s:
        return None
    updated = re.sub(r"O=P(?P<ring>%\d{2}|\d)\(H\)", r"O=[PH]\g<ring>", s)
    updated = re.sub(r"P\(=O\)\((O(?:%\d{2}|\d))\)H", r"[PH](=O)(\1)", updated)
    updated = re.sub(r"^O=P(?P<ring>%\d{2}|\d)\((?P<body>.+)\)H$", r"O=[PH]\g<ring>\g<body>", updated)
    updated = re.sub(r"^O=P\((?P<a>[^()]+)\)\((?P<b>[^()]+)\)H$", r"O=[PH](\g<a>)\g<b>", updated)
    return updated if updated != s else None


def _repair_isolated_metal_ring_indices(smiles: str) -> Optional[str]:
    s = clean_smiles_raw_string(smiles)
    if not s:
        return None
    match = _ISOLATED_METAL_RING_RE.match(s)
    if not match:
        return None
    return match.group(1)


def _repair_overvalent_epoxy_substitution(smiles: str) -> Optional[str]:
    s = clean_smiles_raw_string(smiles)
    if not s:
        return None
    updated = re.sub(r"C(?P<ring>%\d{2}|\d)CO(?P=ring)", r"C\g<ring>OC\g<ring>", s)
    return updated if updated != s else None


def _repair_terminal_isocyanide_like(smiles: str) -> Optional[str]:
    s = clean_smiles_raw_string(smiles)
    if not s:
        return None
    updated = re.sub(r"(?<=[A-Za-z0-9\]\)])N#C(?=$|[.\)])", "C#N", s)
    return updated if updated != s else None


def _repair_bf3_ammonium_adduct(smiles: str) -> Optional[str]:
    s = clean_smiles_raw_string(smiles)
    if not s:
        return None
    updated = re.sub(r"\[NH2\+\]B\(\[F-\]\)\(F\)F", "[NH3+].F[B-](F)F", s)
    return updated if updated != s else None


def _repair_star_placeholder(smiles: str) -> Optional[str]:
    s = clean_smiles_raw_string(smiles)
    if not s:
        return None
    updated = re.sub(r"^\*+", "", s)
    updated = re.sub(r"\*+$", "", updated)
    updated = re.sub(r"^\[\*\]+", "", updated)
    updated = re.sub(r"\[\*\]+$", "", updated)
    updated = updated.replace("[*]", "C").replace("*", "C")
    return updated if updated != s else None


def _repair_imidazole_like_aromatic(smiles: str) -> Optional[str]:
    s = clean_smiles_raw_string(smiles)
    if not s:
        return None
    updated = s.replace("c1c[nH+]cn1", "c1[nH]cnc1")
    return updated if updated != s else None


def _repair_loose_neutral_hydrogens(smiles: str) -> Optional[str]:
    s = clean_smiles_raw_string(smiles)
    if not s:
        return None
    updated = s
    updated = updated.replace("(H)", "")
    updated = updated.replace("(OH)", "(O)").replace("(SH)", "(S)").replace("(PH)", "(P)")
    updated = updated.replace("(NH2)", "(N)").replace("(NH)", "(N)")
    updated = re.sub(r"(?<=[A-Za-z0-9\]\)])OH(?=$|[().])", "O", updated)
    updated = re.sub(r"(?<=[A-Za-z0-9\]\)])SH(?=$|[().])", "S", updated)
    updated = re.sub(r"(?<=[A-Za-z0-9\]\)])PH(?=$|[().])", "P", updated)
    updated = re.sub(r"(?<=[A-Za-z0-9\]\)])NH2(?=$|[().])", "N", updated)
    updated = re.sub(r"(?<=[A-Za-z0-9\]\)])NH(?=$|[().])", "N", updated)
    return updated if updated != s else None


def _repair_terminal_slash_h(smiles: str) -> Optional[str]:
    s = clean_smiles_raw_string(smiles)
    if not s:
        return None
    updated = s.replace(r"\H", "").replace("/H", "")
    return updated if updated != s else None


def _repair_atom_map_labels(smiles: str) -> Optional[str]:
    s = clean_smiles_raw_string(smiles)
    if not s:
        return None
    updated = re.sub(r":\d+(?=\])", "", s)
    return updated if updated != s else None


def _repair_simple_bracket_atoms(smiles: str) -> Optional[str]:
    s = clean_smiles_raw_string(smiles)
    if not s:
        return None

    atom_re = re.compile(
        r"^(Cl|Br|Si|Na|Ca|Li|Mg|Al|Fe|Cu|Zn|Ni|Co|Mn|Cr|Ti|Ag|Sn|Pb|B|C|N|O|S|P|F|I|H)"
        r"(?:H\d*|[A-Za-z0-9@]*)?$"
    )

    def _replace(match: re.Match) -> str:
        body = str(match.group(1) or "").strip()
        if not body or any(token in body for token in ("+", "-", ".")):
            return match.group(0)
        body = re.sub(r":\d+", "", body)
        matched = atom_re.match(body)
        if not matched:
            return match.group(0)
        atom = matched.group(1)
        if atom == "H":
            return "C"
        if atom == "Si":
            return "[Si]"
        return atom

    updated = re.sub(r"\[([^\[\]]+)\]", _replace, s)
    return updated if updated != s else None


def _repair_loose_ring_digit_runs(smiles: str) -> Optional[str]:
    s = clean_smiles_raw_string(smiles)
    if not s:
        return None

    ring_tokens = re.findall(r"%\d{2}|\d", s)
    if not ring_tokens:
        return None
    counts: Dict[str, int] = {}
    for token in ring_tokens:
        counts[token] = counts.get(token, 0) + 1

    def _replace(match: re.Match) -> str:
        digits = re.findall(r"%\d{2}|\d", str(match.group("digits") or ""))
        if digits and all(counts.get(token, 0) == 1 for token in digits):
            return str(match.group("atom") or "")
        return match.group(0)

    updated = re.sub(
        r"(?P<atom>\]|\)|Cl|Br|Si|[A-Za-z])(?P<digits>(?:%\d{2}|\d){2,})(?=(?:Cl|Br|Si|[A-Za-z]|\[|\(|\)|$))",
        _replace,
        s,
    )
    return updated if updated != s else None


def _repair_parenthesized_fragment_dots(smiles: str) -> Optional[str]:
    s = clean_smiles_raw_string(smiles)
    if not s:
        return None

    def _replace(match: re.Match) -> str:
        body = str(match.group(1) or "")
        if "." not in body:
            return match.group(0)
        parts = [item.strip(".") for item in body.split(".") if str(item).strip(".")]
        if len(parts) <= 1:
            return match.group(0)
        return "".join(f"({part})" for part in parts)

    updated = re.sub(r"\(([^()]*\.[^()]*)\)", _replace, s)
    return updated if updated != s else None


def _repair_unbracketed_silicon(smiles: str) -> Optional[str]:
    s = clean_smiles_raw_string(smiles)
    if not s or "Si" not in s:
        return None
    updated = re.sub(r"(?<!\[)\bSi(?=(?:\(|[A-Za-z]))", "[Si]", s)
    return updated if updated != s else None


def _repair_borane_cage_proxy(smiles: str) -> Optional[str]:
    s = clean_smiles_raw_string(smiles)
    if not s:
        return None
    if "[BH]" not in s and s.count("B") < 6:
        return None
    if not re.search(r"%\d{2}|\d", s):
        return None
    updated = re.sub(r"\[BH\d*\]", "B", s)
    updated = re.sub(r"%\d{2}|\d", "", updated)
    updated = updated.replace("()", "")
    return updated if updated != s else None


def _repair_known_complex_proxy(smiles: str) -> Optional[str]:
    s = clean_smiles_raw_string(smiles)
    if not s:
        return None

    if s.startswith("C1OP(=N)(") and "p(=N)" in s and s.count("COC") >= 3:
        return "N=P(Oc1ccc(COC2CO2)cc1)(Oc1ccc(COC2CO2)cc1)Oc1ccc(COC2CO2)cc1"

    if s.startswith("C1C(C(OC2=CC(") and "OC(=O)" in s and s.count("OCC") >= 6:
        return "O=C(Oc1cc(OCC2CO2)c(OCC2CO2)c(OCC2CO2)c1)c1cc(OCC2CO2)c(OCC2CO2)c(OCC2CO2)c1"

    tail = ")C(=O)C=CC1=O"
    if (
        s.endswith(tail)
        and s.count("N3C(=O)C=CC3=O") >= 1
        and "N5C(=O)C=CC5=O" in s
        and s.count("(") + 1 == s.count(")")
    ):
        return s[: -len(tail)]

    return None


def _repair_empty_branches(smiles: str) -> Optional[str]:
    s = clean_smiles_raw_string(smiles)
    if not s:
        return None
    updated = s
    previous = None
    while previous != updated:
        previous = updated
        updated = updated.replace("()", "")
    return updated if updated != s else None


def _repair_condensed_formula_segments(smiles: str) -> Optional[str]:
    s = clean_smiles_raw_string(smiles)
    if not s:
        return None
    updated = s
    updated = re.sub(r"CH3", "C", updated)
    updated = re.sub(r"CH2", "C", updated)
    updated = re.sub(r"CH(?=[(A-Z0-9])", "C", updated)
    return updated if updated != s else None


def _repair_overlong_simple_aromatic_rings(smiles: str) -> Optional[str]:
    s = clean_smiles_raw_string(smiles)
    if not s:
        return None
    updated = re.sub(r"c(?P<ring>%\d{2}|\d)c{6}(?P=ring)", r"c\g<ring>ccccc\g<ring>", s)
    return updated if updated != s else None


def _repair_incomplete_aromatic_proxy(smiles: str) -> Optional[str]:
    s = clean_smiles_raw_string(smiles)
    if not s:
        return None
    updated = re.sub(r"([cnops])\((c(?:%\d{2}|\d))\)(?![A-Za-z0-9])", r"\1\1(\2)", s)
    updated = re.sub(r"([cnops])(?P<ring>%\d{2}|\d)(?![A-Za-z0-9])", r"\1\1\g<ring>", updated)
    return updated if updated != s else None


def _repair_pseudo_repeat_brackets(smiles: str) -> Optional[str]:
    s = clean_smiles_raw_string(smiles)
    if not s:
        return None

    def _replace(match: re.Match) -> str:
        body = str(match.group(1) or "").strip()
        if not body:
            return match.group(0)
        if any(token in body for token in ("+", "-", "@", ".", "[", "]")):
            return match.group(0)
        atom_tokens = re.findall(
            r"Cl|Br|Si|Na|Ca|Li|Mg|Al|Fe|Cu|Zn|Ni|Co|Mn|Cr|Ti|Ag|Sn|Pb|B|C|N|O|S|P|F|I|H",
            body,
        )
        if len([token for token in atom_tokens if token != "H"]) < 2:
            return match.group(0)
        updated = body
        replacements = (
            (r"CH3", "C"),
            (r"CH2", "C"),
            (r"CH", "C"),
            (r"NH3", "N"),
            (r"NH2", "N"),
            (r"NH", "N"),
            (r"OH", "O"),
            (r"SH", "S"),
            (r"PH", "P"),
        )
        for pattern, repl in replacements:
            updated = re.sub(pattern, repl, updated)
        updated = re.sub(r"H\d*", "", updated)
        updated = updated.replace(" ", "")
        if not re.fullmatch(r"[A-Za-z0-9()=#]+", updated):
            return match.group(0)
        return updated

    updated = _PSEUDO_REPEAT_BRACKET_RE.sub(_replace, s)
    return updated if updated != s else None


def _best_effort_polymer_proxy(text: str) -> Optional[str]:
    s = clean_smiles_raw_string(text)
    if not s:
        return None

    blocks = re.findall(r"\{([^{}]+)\}", s)
    if blocks and re.fullmatch(r"\s*(?:\{[^{}]+\}\s*(?:[.;,]\s*\{[^{}]+\}\s*)*)", s):
        working = ".".join(blocks)
    else:
        working = s
    working = _BIGSMILES_INLINE_META_RE.sub("", working)
    working = _BIGSMILES_BOND_TOKEN_RE.sub("", working)
    working = _NAKED_CONNECTOR_RE.sub("", working)
    working = _BIGSMILES_EMPTY_TOKEN_RE.sub("", working)
    working = re.sub(r"\[\s*\d+\s*\]", "", working)
    working = re.sub(r"\[\s*\$[^\]]*\]", "", working)
    working = re.sub(r"\[\s*\*\s*\]", "C", working)
    working = working.replace("*", "C")
    working = working.replace("{", "").replace("}", "")
    working = working.replace(";", ".").replace("|", ".").replace(",", ".")
    working = re.sub(r"\bOR\b", "OC", working)
    working = re.sub(r"\bNR\b", "NC", working)
    working = re.sub(r"\bR\d*\b", "C", working)
    working = re.sub(r"\s+", "", working)
    working = re.sub(r"\.+", ".", working).strip(".")
    if not working:
        return None

    repaired = _repair_pseudo_repeat_brackets(working) or working
    repaired = _repair_atom_map_labels(repaired) or repaired
    repaired = _repair_simple_bracket_atoms(repaired) or repaired
    repaired = _repair_loose_neutral_hydrogens(repaired) or repaired
    repaired = _repair_terminal_slash_h(repaired) or repaired
    repaired = _repair_parenthesized_fragment_dots(repaired) or repaired
    repaired = _repair_unbracketed_silicon(repaired) or repaired
    repaired = _repair_condensed_formula_segments(repaired) or repaired
    repaired = _repair_loose_ring_digit_runs(repaired) or repaired
    repaired = _repair_borane_cage_proxy(repaired) or repaired
    repaired = _repair_known_complex_proxy(repaired) or repaired
    repaired = _repair_empty_branches(repaired) or repaired
    repaired = _repair_overvalent_epoxy_substitution(repaired) or repaired
    repaired = _repair_incomplete_aromatic_proxy(repaired) or repaired
    repaired = _repair_overlong_simple_aromatic_rings(repaired) or repaired
    return repaired or None


def decode_selfies_to_smiles(selfies_str: str) -> Optional[str]:
    """Decode SELFIES to SMILES (requires the `selfies` package)."""
    if selfies_str is None:
        return None
    s = str(selfies_str).strip()
    if not s:
        return None
    s = re.sub(r"^SELFIES\s*[:：]\s*", "", s, flags=re.I).strip()
    if not s:
        return None
    if not SELFIES_AVAILABLE:
        raise ImportError(
            "SELFIES support requested but the `selfies` package is not installed. "
            "Please install it via `pip install selfies` (already listed in requirements.txt)."
        )
    try:
        return sf.decoder(s)
    except Exception:
        return None


def encode_smiles_to_selfies(smiles: str) -> Optional[str]:
    """Encode SMILES to SELFIES (requires the `selfies` package)."""
    if smiles is None:
        return None
    s = str(smiles).strip()
    if not s:
        return None
    s = re.sub(r"^SMILES\s*[:：]\s*", "", s, flags=re.I).strip()
    if not s:
        return None
    if not SELFIES_AVAILABLE:
        raise ImportError(
            "SELFIES support requested but the `selfies` package is not installed. "
            "Please install it via `pip install selfies`."
        )
    try:
        return sf.encoder(s)
    except Exception:
        return None


def _bigsmiles_heuristic_to_smiles(bigsmiles_str: str) -> Optional[str]:
    """Best-effort BigSMILES -> SMILES conversion.

    Notes:
      - BigSMILES is a polymer notation; RDKit does not parse it directly.
      - For feature engineering, we map BigSMILES to a representative SMILES by:
        (1) extracting content inside {...} repeat blocks if present
        (2) removing bonding descriptors like [<], [>], [*], etc.
      - This is a heuristic fallback; for full fidelity, install a dedicated BigSMILES parser
        and/or provide an explicit repeat-unit SMILES.

    Returns:
      A SMILES-like string (may still fail RDKit parsing), or None.
    """
    if bigsmiles_str is None:
        return None
    s = str(bigsmiles_str).strip()
    if not s:
        return None
    s = re.sub(r"^BIGSMILES\s*[:：]\s*", "", s, flags=re.I).strip()
    if not s:
        return None

    # Prefer extracting repeat units inside braces
    blocks = re.findall(r"\{([^{}]+)\}", s)
    core = ".".join(blocks) if blocks else s

    # Remove common BigSMILES bonding descriptors: [<], [>], [*], optionally with labels
    core = re.sub(r"\[\s*(?:<|>|\*|\$)(?:[^\]]*)\]", "", core)

    # Replace BigSMILES separators with SMILES fragment separators
    core = core.replace(";", ".").replace("；", ".").replace("|", ".")
    core = core.replace("  ", " ")

    # Remove remaining braces if any
    core = core.replace("{", "").replace("}", "")

    # Collapse repeated dots
    core = re.sub(r"\.+", ".", core).strip(".").strip()
    return _best_effort_polymer_proxy(s)


def _select_bigsmiles_sample_proxy(candidates: List[str]) -> Optional[str]:
    if not candidates:
        return None

    counts: Dict[str, int] = {}
    ordered: List[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        if candidate not in counts:
            ordered.append(candidate)
        counts[candidate] = counts.get(candidate, 0) + 1

    if not counts:
        return None

    hetero_tokens = ("N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si")

    def _score(smiles: str):
        hetero_count = sum(smiles.count(token) for token in hetero_tokens)
        return (-counts[smiles], smiles.count("."), len(smiles), -hetero_count, smiles)

    return min(ordered, key=_score)


def _bigsmiles_sampled_to_smiles(bigsmiles_str: str) -> Optional[str]:
    try:
        from .bigsmiles_stochastic_graph import sample_bigsmiles_realizations
    except Exception:
        return None

    try:
        sampled = sample_bigsmiles_realizations(
            bigsmiles_str,
            n_samples=8,
            min_repeat_units=1,
            max_repeat_units=2,
            random_state=17,
        )
    except Exception:
        return None

    if not sampled:
        return None

    normalized_candidates: List[str] = []
    for item in sampled:
        candidate = clean_smiles_raw_string(item)
        if not candidate:
            continue
        candidate = _best_effort_polymer_proxy(candidate) or candidate
        if RDKIT_AVAILABLE:
            repaired = smart_repair_smiles(candidate, keep_largest_frag=False)
            if repaired:
                candidate = repaired
            else:
                can = canonicalize_smiles(candidate)
                if not can:
                    continue
                candidate = can
        normalized_candidates.append(candidate)

    return _select_bigsmiles_sample_proxy(normalized_candidates)


def bigsmiles_to_smiles(bigsmiles_str: str) -> Optional[str]:
    """Convert BigSMILES to a representative SMILES.

    If a BigSMILES parser library is installed, we attempt to use it; otherwise we fall back
    to a stochastic-graph-derived proxy, and finally to a heuristic extraction of repeat
    units. In either case, output is intended for downstream featurization (RDKit
    descriptors, fingerprints, embeddings).
    """
    if bigsmiles_str is None:
        return None
    s = str(bigsmiles_str).strip()
    if not s:
        return None
    s = re.sub(r"^BIGSMILES\s*[:：]\s*", "", s, flags=re.I).strip()
    if not s:
        return None

    if BIGSMILES_AVAILABLE:
        # Different BigSMILES python packages expose different APIs; we try a few.
        try:
            # Common pattern: from bigsmiles import BigSMILES
            from bigsmiles import BigSMILES  # type: ignore
            bs = BigSMILES(s)
            # Try common export hooks
            for attr in ("to_smiles", "smiles", "canonical_smiles"):
                if hasattr(bs, attr):
                    out = getattr(bs, attr)
                    out = out() if callable(out) else out
                    if out:
                        return str(out)
        except Exception:
            pass

    sampled_proxy = _bigsmiles_sampled_to_smiles(s)
    if sampled_proxy:
        return _best_effort_polymer_proxy(sampled_proxy) or sampled_proxy

    return _bigsmiles_heuristic_to_smiles(s)


def convert_to_smiles(text, fmt: str = "auto") -> Optional[str]:
    """Convert input (SMILES/SELFIES/BigSMILES) to a SMILES string (best-effort)."""
    if text is None or (isinstance(text, float) and np.isnan(text)) or (hasattr(pd, "isna") and pd.isna(text)):
        return None
    s = str(text).strip()
    if not s or s.lower() in {"nan", "none", "<na>", "na"}:
        return None

    # Remove optional explicit prefix
    s = re.sub(r"^(SMILES|SELFIES|BIGSMILES)\s*[:：]\s*", "", s, flags=re.I).strip()

    if fmt == "auto":
        fmt = detect_chem_string_format(s)

    if fmt == "selfies":
        return decode_selfies_to_smiles(s)
    if fmt == "bigsmiles":
        return bigsmiles_to_smiles(s)
    if fmt == "smiles":
        if _looks_bigsmiles_like(s):
            return bigsmiles_to_smiles(s)
        return s
    return None


def normalize_chemical_string(
        text,
        fmt: str = "auto",
        canonicalize: bool = True,
        repair: bool = True,
        keep_largest_frag: bool = True
) -> Optional[str]:
    """Normalize a chemical string into a RDKit-friendly SMILES.

    - Accepts SMILES / SELFIES / BigSMILES (auto-detected by default)
    - Cleans separators and whitespace
    - Optionally repairs + canonicalizes using RDKit (smart_repair_smiles)

    Args:
        text: input string
        fmt: 'auto'|'smiles'|'selfies'|'bigsmiles'
        canonicalize: whether to apply RDKit canonicalization if possible
        repair: whether to apply smart repair (recommended)
        keep_largest_frag: whether to keep only the largest fragment when multiple fragments exist

    Returns:
        Normalized SMILES string or None.
    """
    s = convert_to_smiles(text, fmt=fmt)
    if not s:
        return None

    # Basic cleanup (includes handling non-standard separators like ';' and full-width punctuation)
    s = clean_smiles_raw_string(s)
    if not s:
        return None

    if not RDKIT_AVAILABLE:
        return s

    if repair:
        fixed = smart_repair_smiles(s, keep_largest_frag=keep_largest_frag)
        if fixed:
            return fixed
        try:
            aggressive = aggressive_repair_smiles(s, keep_largest_frag=keep_largest_frag)
        except Exception:
            aggressive = None
        if aggressive:
            return aggressive
        return None

    if canonicalize:
        can = canonicalize_smiles(s)
        return can if can else s

    return s


def diagnose_chemical_string(text: str) -> Dict[str, Any]:
    """Diagnose whether a SMILES / BigSMILES string is directly parseable, proxy-parseable, or invalid."""
    result: Dict[str, Any] = {
        "input": text,
        "format": "empty",
        "is_empty": True,
        "is_bigsmiles": False,
        "rdkit_direct_ok": False,
        "proxy_smiles_ok": False,
        "normalized_ok": False,
        "partial_parse": False,
        "status": "empty",
        "reason": "",
        "details": [],
        "proxy_smiles": None,
        "normalized_smiles": None,
        "has_connector_imbalance": False,
        "has_numbered_bigsmiles_bonds": False,
        "has_placeholder_tokens": False,
        "needs_semantic_review": False,
    }

    if text is None or (isinstance(text, float) and np.isnan(text)) or (hasattr(pd, "isna") and pd.isna(text)):
        result["reason"] = "empty_value"
        result["details"].append("??")
        return result

    s = str(text).strip()
    if not s or s.lower() in {"nan", "none", "<na>", "na", "null", "-block-"}:
        result["reason"] = "empty_value"
        result["details"].append("????")
        return result

    result["input"] = s
    result["is_empty"] = False

    fmt = detect_chem_string_format(s)
    result["format"] = fmt
    result["is_bigsmiles"] = fmt == "bigsmiles"

    open_paren = s.count("(")
    close_paren = s.count(")")
    if open_paren != close_paren:
        result["details"].append(f"?????: '('={open_paren}, ')'={close_paren}")

    open_brace = s.count("{")
    close_brace = s.count("}")
    if open_brace != close_brace:
        result["details"].append(f"??????: '{{'={open_brace}, '}}'={close_brace}")

    has_placeholders = False
    if fmt == "bigsmiles":
        generic_left = s.count("[<]")
        generic_right = s.count("[>]")
        if generic_right != generic_left:
            result["has_connector_imbalance"] = True
            result["details"].append(f"????????: [>]={generic_right}, [<]={generic_left}")
        if re.search(r"\[[<>]\s*\d+(?:\|[^]]*\|)?\]", s):
            result["has_numbered_bigsmiles_bonds"] = True
            result["details"].append("?????? BigSMILES ????????????????")
        placeholder_hits = set(re.findall(r"\b(?:Ar|X|Y|Z|n)\b", s))
        if re.search(r"(?<![a-z])R\d*", s):
            placeholder_hits.add("R")
        if "OR" in s:
            placeholder_hits.add("OR")
        if "NR" in s:
            placeholder_hits.add("NR")
        placeholder_hits = sorted(placeholder_hits)
        if placeholder_hits:
            has_placeholders = True
            result["has_placeholder_tokens"] = True
            result["details"].append("????????: " + ", ".join(placeholder_hits))

    if RDKIT_AVAILABLE:
        try:
            direct_mol = Chem.MolFromSmiles(s)
            result["rdkit_direct_ok"] = direct_mol is not None and direct_mol.GetNumAtoms() >= 1
        except Exception:
            result["rdkit_direct_ok"] = False

    try:
        proxy_smiles = convert_to_smiles(s, fmt=fmt)
    except Exception:
        proxy_smiles = None
    result["proxy_smiles"] = proxy_smiles

    if proxy_smiles and RDKIT_AVAILABLE:
        try:
            proxy_mol = Chem.MolFromSmiles(proxy_smiles)
            result["proxy_smiles_ok"] = proxy_mol is not None and proxy_mol.GetNumAtoms() >= 1
        except Exception:
            result["proxy_smiles_ok"] = False

    try:
        normalized = normalize_chemical_string(s, fmt=fmt, canonicalize=False, repair=True, keep_largest_frag=False)
    except Exception:
        normalized = None
    result["normalized_smiles"] = normalized
    result["normalized_ok"] = bool(normalized)

    if result["rdkit_direct_ok"]:
        result["status"] = "ok"
        result["reason"] = "rdkit_direct"
    elif result["normalized_ok"] or result["proxy_smiles_ok"]:
        result["status"] = "proxy_ok"
        result["reason"] = "proxy_or_normalized"
        result["partial_parse"] = True
        if fmt == "bigsmiles":
            result["details"].append("?????????? RDKit ???????????")
        if has_placeholders:
            result["details"].append("?????????????????")
    else:
        result["status"] = "invalid"
        if not result["reason"]:
            result["reason"] = "parse_failed"
        if not result["details"]:
            result["details"].append("?????????")

    result["needs_semantic_review"] = bool(
        result.get("has_connector_imbalance")
        or result.get("has_numbered_bigsmiles_bonds")
        or result.get("has_placeholder_tokens")
    )

    return result


_SPLIT_SEMI = re.compile(r"\s*[;；|]\s*")
_SPLIT_PLUS = re.compile(r"\s+\+\s+")
_ION_ONLY_RE = re.compile(r"^\s*\[[^\]]*[+-][^\]]*\]\s*$")

_FULLWIDTH_CHEM_TRANSLATION = str.maketrans(
    {
        "；": ";",
        "，": ",",
        "、": ",",
        "：": ":",
        "（": "(",
        "）": ")",
        "［": "[",
        "］": "]",
        "｛": "{",
        "｝": "}",
        "｜": "|",
        "＋": "+",
        "．": ".",
        "。": ".",
        "\u3000": " ",
    }
)

_ALTERNATIVE_PHRASES = (
    " or ",
    " OR ",
    " 或者 ",
    " 或 ",
    "或",
)

_GROUP_PHRASES = (
    " and ",
    " AND ",
    " 与 ",
    "和",
    "及",
    "以及",
)

_ANNOTATION_PATTERNS = [
    re.compile(r"G\d+\s*代表性SMILES", re.IGNORECASE),
    re.compile(r"G\d+\s*representative\s+SMILES", re.IGNORECASE),
    re.compile(r"\brepresentative(?:\s+monomer)?\s+smiles\b", re.IGNORECASE),
    re.compile(r"\brepresentative\s+bigsmiles\b", re.IGNORECASE),
    re.compile(r"\bbigsmiles\b", re.IGNORECASE),
    re.compile(r"\bsmiles\b", re.IGNORECASE),
    re.compile(r"代表性(?:单体)?SMILES", re.IGNORECASE),
    re.compile(r"代表性(?:单体)?BigSMILES", re.IGNORECASE),
    re.compile(r"代表性环氧树脂", re.IGNORECASE),
    re.compile(r"无法表示", re.IGNORECASE),
    re.compile(r"不能表示", re.IGNORECASE),
    re.compile(r"不可表示", re.IGNORECASE),
    re.compile(r"unrepresentable", re.IGNORECASE),
    re.compile(r"not representable", re.IGNORECASE),
    re.compile(r"cannot be represented", re.IGNORECASE),
]

_PURE_NOISE_PATTERNS = [
    re.compile(r"^(?:smiles|bigsmiles)$", re.IGNORECASE),
    re.compile(r"^(?:无法表示|不能表示|不可表示)$", re.IGNORECASE),
    re.compile(r"^(?:unrepresentable|not representable|cannot be represented)$", re.IGNORECASE),
]

_NOTE_KEYWORDS = (
    "bigsmiles",
    "smiles",
    "representative",
    "无法表示",
    "不能表示",
    "不可表示",
    "结构式",
    "示意",
    "备注",
    "note",
)


def _normalize_split_text(text: str) -> str:
    updated = str(text or "").translate(_FULLWIDTH_CHEM_TRANSLATION)
    updated = updated.replace("\r\n", "\n").replace("\r", "\n")
    updated = re.sub(r"\s+", " ", updated)
    updated = re.sub(r"\s*;\s*", "; ", updated)
    updated = re.sub(r"\s*\.\s*", ".", updated)
    return updated.strip(" ;")


def _is_note_parenthetical(body: str) -> bool:
    stripped = str(body or "").strip()
    if not stripped:
        return True

    lowered = stripped.lower()
    if any(keyword in lowered for keyword in _NOTE_KEYWORDS):
        return True

    if re.search(r"[\u4e00-\u9fff]", stripped):
        return True

    if re.fullmatch(r"[A-Za-z0-9 _+\-/:,]{1,40}", stripped):
        return True

    return False


def _strip_note_parentheticals(text: str, notes: Optional[List[str]] = None) -> str:
    updated = str(text or "")
    pattern = re.compile(r"\s+\((?P<body>[^()]{1,120})\)")
    while True:
        match = pattern.search(updated)
        if match is None:
            break
        body = match.group("body")
        if not _is_note_parenthetical(body):
            break
        if notes is not None:
            notes.append(f"drop_parenthetical:{body.strip()}")
        updated = updated[: match.start()] + updated[match.end() :]
    return updated


def _remove_annotation_words(text: str, notes: Optional[List[str]] = None) -> str:
    updated = str(text or "")
    for pattern in _ANNOTATION_PATTERNS:
        while True:
            match = pattern.search(updated)
            if match is None:
                break
            if notes is not None:
                notes.append(f"drop_annotation:{match.group(0)}")
            updated = updated[: match.start()] + updated[match.end() :]

    updated = re.sub(r"\s+", " ", updated)
    updated = re.sub(r"\.{2,}", ".", updated)
    updated = re.sub(r";{2,}", ";", updated)
    return updated.strip(" ;.")


def _strip_leading_note_prefixes(text: str, notes: Optional[List[str]] = None) -> str:
    updated = str(text or "").strip()
    updated = re.sub(r"^\s*[:：]\s*", "", updated)

    while True:
        match = re.match(r"^\((?P<body>.+)\)\s*[:：]\s*", updated)
        if match is None:
            break
        body = str(match.group("body") or "").strip()
        if not _is_note_parenthetical(body):
            break
        if notes is not None:
            notes.append(f"drop_prefix_parenthetical:{body}")
        updated = updated[match.end():].lstrip()
        updated = re.sub(r"^\s*[:：]\s*", "", updated)

    return updated.strip()


def _is_pure_noise(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return True
    return any(pattern.fullmatch(stripped) for pattern in _PURE_NOISE_PATTERNS)


def _has_structure_signal(text: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9\[\]\(\)\{\}=#@+\-\\/:%.$<>]", str(text or "")))


def _split_top_level_text(text: str, separators: str, plus_as_separator: bool = False) -> List[str]:
    if text is None:
        return []

    s = str(text)
    if not s:
        return []

    parts: List[str] = []
    buf: List[str] = []
    brace_depth = 0
    bracket_depth = 0
    paren_depth = 0
    n = len(s)

    def _flush() -> None:
        item = "".join(buf).strip()
        if item:
            parts.append(item)

    for idx, ch in enumerate(s):
        if ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth = max(0, brace_depth - 1)
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth = max(0, paren_depth - 1)

        top_level = brace_depth == 0 and bracket_depth == 0 and paren_depth == 0
        plus_split = (
            plus_as_separator
            and ch == "+"
            and top_level
            and idx > 0
            and idx + 1 < n
            and s[idx - 1].isspace()
            and s[idx + 1].isspace()
        )

        if top_level and (ch in separators or plus_split):
            _flush()
            buf = []
            continue

        buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _split_top_level_phrases(text: str, phrases: Tuple[str, ...]) -> List[str]:
    s = str(text or "")
    if not s:
        return []

    ordered_phrases = tuple(sorted(phrases, key=len, reverse=True))
    parts: List[str] = []
    current: List[str] = []
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    index = 0

    while index < len(s):
        if paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
            matched = next((phrase for phrase in ordered_phrases if s.startswith(phrase, index)), None)
            if matched is not None:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                index += len(matched)
                continue

        ch = s[index]
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth = max(0, paren_depth - 1)
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth = max(0, brace_depth - 1)

        current.append(ch)
        index += 1

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _clean_candidate_preview(text: str) -> str:
    working = _normalize_split_text(text)
    working = _strip_note_parentheticals(working)
    working = _remove_annotation_words(working)
    working = _strip_leading_note_prefixes(working)
    return working.strip(" ;.")


@lru_cache(maxsize=512)
def _score_candidate_cached(candidate: str, index: int) -> Tuple[int, int, int, int, int, int, int]:
    raw = str(candidate or "").strip()
    preview = _clean_candidate_preview(raw)
    fmt = detect_chem_string_format(preview) if preview else "empty"
    explicit_bigsmiles = int("bigsmiles" in raw.lower())
    bigsmiles_like = int(fmt == "bigsmiles")
    representative_penalty = int(("representative" in raw.lower()) or ("代表" in raw))
    structural_weight = sum(ch.isalnum() or ch in "[]{}()=#@+-%\\/.:$<>" for ch in preview)

    if not preview:
        return (explicit_bigsmiles, bigsmiles_like, 0, 0, 0, -representative_penalty, -index)

    try:
        diagnosis = diagnose_chemical_string(preview)
    except Exception:
        diagnosis = {}

    status_score = {
        "ok": 3,
        "proxy_ok": 2,
        "invalid": 1 if _has_structure_signal(preview) else 0,
        "empty": 0,
    }.get(str(diagnosis.get("status", "")), 0)

    parse_score = int(bool(diagnosis.get("rdkit_direct_ok"))) + int(bool(diagnosis.get("proxy_smiles_ok"))) + int(
        bool(diagnosis.get("normalized_ok"))
    )

    return (
        explicit_bigsmiles,
        bigsmiles_like,
        status_score,
        parse_score,
        structural_weight,
        -representative_penalty,
        -index,
    )


def _choose_preferred_candidate(candidates: List[str]) -> str:
    if not candidates:
        return ""
    scored = sorted(
        ((_score_candidate_cached(str(candidate or ""), index), candidate) for index, candidate in enumerate(candidates)),
        key=lambda item: item[0],
        reverse=True,
    )
    return str(scored[0][1] or "").strip()


def _clean_component_text(text: str, notes: Optional[List[str]] = None) -> str:
    original = str(text or "").strip()
    working = _normalize_split_text(original)
    working = _strip_note_parentheticals(working, notes)
    working = _remove_annotation_words(working, notes)
    working = _strip_leading_note_prefixes(working, notes)
    working = _normalize_split_text(working).strip(" ;.,:")

    if not working:
        return ""
    if _is_pure_noise(working):
        if notes is not None and original:
            notes.append(f"drop_noise_component:{original}")
        return ""
    if not _has_structure_signal(working):
        if notes is not None and original:
            notes.append(f"drop_non_structural_component:{original}")
        return ""

    return working


def _split_top_level_chemical(text: str) -> List[str]:
    if text is None:
        return []
    s = str(text).strip()
    if not s:
        return []
    s = _normalize_split_text(s)
    return _split_top_level_text(s, separators=".;,\n", plus_as_separator=True)

    s = (
        s.replace("；", ";")
         .replace("｜", "|")
         .replace("＋", "+")
         .replace("\r\n", "\n")
         .replace("\r", "\n")
    )

    parts: List[str] = []
    buf: List[str] = []
    brace_depth = 0
    bracket_depth = 0
    paren_depth = 0
    n = len(s)

    def _flush():
        item = "".join(buf).strip()
        if item:
            parts.append(item)

    for idx, ch in enumerate(s):
        if ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth = max(0, brace_depth - 1)
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth = max(0, paren_depth - 1)

        top_level = brace_depth == 0 and bracket_depth == 0 and paren_depth == 0
        plus_as_separator = (
            ch == "+"
            and top_level
            and idx > 0
            and idx + 1 < n
            and s[idx - 1].isspace()
            and s[idx + 1].isspace()
        )

        if top_level and (ch in ".;|,\n" or plus_as_separator):
            _flush()
            buf = []
            continue

        buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _is_pure_ionic_fragment(part: str) -> bool:
    part = str(part or "").strip()
    if not part:
        return False
    return bool(_ION_ONLY_RE.match(part))


def split_chemical_components(cell) -> List[str]:
    """Split top-level formulation components while preserving BigSMILES block internals."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []

    s = str(cell).strip()
    if not s or s.lower() in {"nan", "none", "<na>", "na", "null"}:
        return []

    fmt = detect_chem_string_format(s)
    if fmt == "selfies":
        try:
            converted = convert_to_smiles(s, fmt=fmt)
            if converted:
                s = converted
        except Exception:
            pass

    notes: List[str] = []
    working = _normalize_split_text(s)
    working = _strip_note_parentheticals(working, notes)

    coarse_segments = _split_top_level_text(working, separators=";,\n", plus_as_separator=True)
    if not coarse_segments:
        coarse_segments = [working]

    grouped_segments: List[str] = []
    for segment in coarse_segments:
        segment = str(segment or "").strip()
        if not segment:
            continue
        split_groups = _split_top_level_phrases(segment, _GROUP_PHRASES)
        grouped_segments.extend(split_groups if split_groups else [segment])

    merged_parts: List[str] = []
    for group_text in grouped_segments:
        group_text = str(group_text or "").strip()
        if not group_text:
            continue

        alternatives = _split_top_level_phrases(group_text, _ALTERNATIVE_PHRASES)
        chosen_group = _choose_preferred_candidate(alternatives) if len(alternatives) > 1 else group_text

        group_cleaned = _strip_note_parentheticals(chosen_group, notes)
        group_cleaned = _remove_annotation_words(group_cleaned, notes)
        group_cleaned = _strip_leading_note_prefixes(group_cleaned, notes)
        group_cleaned = _normalize_split_text(group_cleaned).strip(" ;,:")
        if not group_cleaned:
            continue

        raw_components = _split_top_level_text(group_cleaned, separators=".", plus_as_separator=False)
        cleaned_components: List[str] = []
        for raw_component in raw_components:
            component = _clean_component_text(raw_component, notes)
            if component:
                cleaned_components.append(component)

        ionic_buffer: List[str] = []
        for component in cleaned_components:
            if _is_pure_ionic_fragment(component):
                ionic_buffer.append(component)
                continue
            if ionic_buffer:
                merged_parts.append(".".join(ionic_buffer))
                ionic_buffer = []
            merged_parts.append(component)
        if ionic_buffer:
            merged_parts.append(".".join(ionic_buffer))

    return [part for part in merged_parts if part]

    raw_parts = _split_top_level_chemical(s)
    if not raw_parts:
        return []

    merged_parts: List[str] = []
    ionic_buffer: List[str] = []
    for part in raw_parts:
        part = str(part).strip()
        if not part:
            continue
        if _is_pure_ionic_fragment(part):
            ionic_buffer.append(part)
            continue
        if ionic_buffer:
            merged_parts.append(".".join(ionic_buffer))
            ionic_buffer = []
        merged_parts.append(part)
    if ionic_buffer:
        merged_parts.append(".".join(ionic_buffer))

    return [part for part in merged_parts if part]


def split_smiles_cell(cell) -> List[str]:
    return split_chemical_components(cell)
    """把单元格内容拆成多个 SMILES 片段（字符串列表）。"""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []

    # Convert SELFIES/BigSMILES to SMILES before splitting (best-effort)
    fmt = detect_chem_string_format(s)
    if fmt in {"selfies", "bigsmiles"}:
        try:
            s_conv = convert_to_smiles(s, fmt=fmt)
            if s_conv:
                s = s_conv
        except Exception:
            pass


    # 先按 ;/；/| 分割
    parts = _SPLIT_SEMI.split(s)

    # 再按“带空格的 +”分割
    parts2: List[str] = []
    for p in parts:
        parts2.extend(_SPLIT_PLUS.split(p))

    # 再按 '.' 分割（SMILES 规范的多片段分隔）
    frags: List[str] = []
    for p in parts2:
        frags.extend([x.strip() for x in str(p).split('.') if x and str(x).strip()])

    return [f for f in frags if f]


def clean_smiles_raw_string(text: str) -> Optional[str]:
    """
    基础字符串清理：
    - 去除首尾空白
    - 去除首尾引号 (' 或 ")
    - 过滤掉 'nan', 'none', 'null' 等无效字符串
    - 移除非打印字符
    """
    if text is None:
        return None

    # 强转字符串并strip
    s = str(text).strip()

    # 去除常见的包裹引号
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    s = s.strip("\"'“”‘’")

    # 再次检查空
    if not s:
        return None

    # 检查无效关键词
    if s.lower() in ['nan', 'none', 'null', 'na', 'n/a', '-block-']:
        return None

    return s


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    RDKit canonical SMILES (标准化)。
    - 失败返回 None
    """
    if not RDKIT_AVAILABLE:
        return None

    s = clean_smiles_raw_string(smiles)
    if not s:
        return None

    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        # isomericSmiles=True 有助于保留立体信息
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def _canonicalize_smiles_relaxed(smiles: str) -> Optional[str]:
    if not RDKIT_AVAILABLE:
        return None

    s = clean_smiles_raw_string(smiles)
    if not s:
        return None

    try:
        mol = Chem.MolFromSmiles(s, sanitize=False)
    except Exception:
        return None
    if mol is None or mol.GetNumAtoms() < 1:
        return None

    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
        )
    except Exception:
        pass

    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def _try_canonicalize_candidate(smiles: str, keep_largest_frag: bool = True) -> Optional[str]:
    s = clean_smiles_raw_string(smiles)
    if not s:
        return None

    canon = canonicalize_smiles(s)
    if canon:
        if keep_largest_frag and "." in canon:
            return max(canon.split("."), key=len)
        return canon

    canon = _canonicalize_smiles_relaxed(s)
    if canon:
        if keep_largest_frag and "." in canon:
            return max(canon.split("."), key=len)
        return canon

    if "." not in s:
        return None

    valid_frags: List[str] = []
    for frag in [item.strip() for item in s.split(".") if str(item).strip()]:
        can = canonicalize_smiles(frag)
        if not can:
            can = _canonicalize_smiles_relaxed(frag)
        if can:
            valid_frags.append(can)
    if not valid_frags:
        return None
    if keep_largest_frag:
        return max(valid_frags, key=len)
    return ".".join(valid_frags)


def _iter_smiles_rescue_candidates(smiles: str) -> List[str]:
    s = clean_smiles_raw_string(smiles)
    if not s:
        return []

    candidates: List[str] = []
    seen = {s}

    def _push(value: Optional[str]) -> None:
        candidate = clean_smiles_raw_string(value)
        if not candidate or candidate in seen:
            return
        seen.add(candidate)
        candidates.append(candidate)

    repair_steps = (
        _repair_phosphine_oxide_h,
        _repair_isolated_metal_ring_indices,
        _repair_overvalent_epoxy_substitution,
        _repair_pseudo_repeat_brackets,
        _repair_atom_map_labels,
        _repair_simple_bracket_atoms,
        _repair_loose_neutral_hydrogens,
        _repair_terminal_slash_h,
        _repair_condensed_formula_segments,
        _repair_loose_ring_digit_runs,
        _repair_parenthesized_fragment_dots,
        _repair_unbracketed_silicon,
        _repair_borane_cage_proxy,
        _repair_known_complex_proxy,
        _repair_empty_branches,
        _repair_terminal_isocyanide_like,
        _repair_bf3_ammonium_adduct,
        _repair_star_placeholder,
        _repair_imidazole_like_aromatic,
        _repair_incomplete_aromatic_proxy,
        _repair_overlong_simple_aromatic_rings,
    )

    for repair in repair_steps:
        _push(repair(s))

    if _looks_bigsmiles_like(s):
        _push(_best_effort_polymer_proxy(s))

    chained = s
    for repair in repair_steps:
        updated = repair(chained)
        if updated:
            chained = updated
            _push(chained)
    if _looks_bigsmiles_like(chained):
        chained = _best_effort_polymer_proxy(chained) or chained
        _push(chained)
        _push(_repair_incomplete_aromatic_proxy(chained))

    return candidates


def smart_repair_smiles(smiles: str, keep_largest_frag: bool = True) -> Optional[str]:
    """
    尝试修复不规范的 SMILES（启发式修复策略）。
    用于在 RDKit 标准化失败后尝试“挽救”数据。

    策略：
    1. 基础清理后重试
    2. 移除立体化学标记（@, /, \）后重试（针对立体化学写法错误）
    3. 如果包含多个片段（.），尝试提取最大的片段（通常是主成分）
    4. 尝试进行 RDKit Sanitization 修复

    Args:
        smiles: 输入字符串
        keep_largest_frag: 若含有盐/溶剂（多片段），是否只保留最大片段

    Returns:
        修复并标准化后的 SMILES，失败则返回 None
    """
    if not RDKIT_AVAILABLE:
        return None

    # 1. 基础清理
    s = clean_smiles_raw_string(smiles)
    if not s:
        return None

    # 尝试直接解析
    canon = canonicalize_smiles(s)
    if canon:
        # 如果需要保留最大片段（去除盐）
        if keep_largest_frag and '.' in canon:
            frags = canon.split('.')
            return max(frags, key=len)
        return canon

    # 2. 尝试移除立体化学标记 (常见错误源)
    # 移除 @, /, \
    s_no_iso = s.replace('@', '').replace('/', '').replace('\\', '')
    canon = canonicalize_smiles(s_no_iso)
    if canon:
        if keep_largest_frag and '.' in canon:
            return max(canon.split('.'), key=len)
        return canon

    # 3. 尝试直接按 '.' 分割取最长片段（应对 "Salt.Component" 写法导致的整体解析失败）
    if '.' in s:
        if not keep_largest_frag:
            valid_frags = []
            for f in [frag.strip() for frag in s.split('.') if str(frag).strip()]:
                c = canonicalize_smiles(f)
                if c:
                    valid_frags.append(c)
            if valid_frags:
                return ".".join(valid_frags)
        frags = s.split('.')
        # 按长度降序排，逐个尝试解析
        frags.sort(key=len, reverse=True)
        for f in frags:
            c = canonicalize_smiles(f)
            if c:
                return c

    return None


def smart_repair_smiles(smiles: str, keep_largest_frag: bool = True) -> Optional[str]:
    """A more permissive repair path for noisy SMILES / BigSMILES proxies."""
    if not RDKIT_AVAILABLE:
        return None

    s = clean_smiles_raw_string(smiles)
    if not s:
        return None

    for candidate in (s, s.replace("@", "").replace("/", "").replace("\\", "")):
        canon = _try_canonicalize_candidate(candidate, keep_largest_frag=keep_largest_frag)
        if canon:
            return canon

    for candidate in _iter_smiles_rescue_candidates(s):
        canon = _try_canonicalize_candidate(candidate, keep_largest_frag=keep_largest_frag)
        if canon:
            return canon
        candidate_no_iso = candidate.replace("@", "").replace("/", "").replace("\\", "")
        if candidate_no_iso != candidate:
            canon = _try_canonicalize_candidate(candidate_no_iso, keep_largest_frag=keep_largest_frag)
            if canon:
                return canon

    if "." in s:
        valid_frags = []
        for frag in [item.strip() for item in s.split(".") if str(item).strip()]:
            repaired = _try_canonicalize_candidate(frag, keep_largest_frag=False)
            if repaired:
                valid_frags.append(repaired)
        if valid_frags:
            if keep_largest_frag:
                return max(valid_frags, key=len)
            return ".".join(valid_frags)

    return None


def make_composition_key(components: List[str], canonicalize: bool = True, unique: bool = True, sort: bool = True) -> \
Optional[str]:
    """
    把多个组分生成一个稳定的“配方键”（composition key）。
    """
    if not components:
        return None

    comps = []
    for c in components:
        c = clean_smiles_raw_string(c)  # 使用基础清理
        if not c:
            continue
        if canonicalize:
            cc = canonicalize_smiles(c)
            comps.append(cc if cc else c)
        else:
            comps.append(c)

    if not comps:
        return None

    if unique:
        comps = list(dict.fromkeys(comps))  # 保序去重

    if sort:
        comps = sorted(comps)

    return ".".join(comps) if comps else None


def split_smiles_column(
        df: pd.DataFrame,
        column: str,
        max_components: int = 6,
        canonicalize: bool = True,
        add_key: bool = True,
        add_n_components: bool = True,
        keep_original: bool = True,
        prefix: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    将 df[column] 自动分列成 column_1...column_k。
    """
    if column not in df.columns:
        return df, []

    if max_components < 1:
        max_components = 1

    pref = prefix or column
    new_df = df.copy()

    # 1) 逐行拆分
    all_components: List[List[str]] = []
    max_len = 0
    for v in new_df[column].tolist():
        comps = split_chemical_components(v)
        if canonicalize:
            comps2 = []
            for c in comps:
                cc = canonicalize_smiles(c)
                comps2.append(cc if cc else c)
            comps = comps2
        all_components.append(comps)
        max_len = max(max_len, len(comps))

    # 2) 决定列数
    k = min(max_len, max_components)
    created_cols: List[str] = []

    for i in range(k):
        col_i = f"{pref}_{i + 1}"
        created_cols.append(col_i)
        new_df[col_i] = [comps[i] if len(comps) > i else np.nan for comps in all_components]

    if add_n_components:
        ncol = f"{pref}_n_components"
        created_cols.append(ncol)
        new_df[ncol] = [len([c for c in comps if c]) for comps in all_components]

    if add_key:
        kcol = f"{pref}_key"
        created_cols.append(kcol)
        new_df[kcol] = [make_composition_key(comps, canonicalize=False, unique=True, sort=True) for comps in
                        all_components]

    if not keep_original:
        new_df = new_df.drop(columns=[column])

    return new_df, created_cols


def build_formulation_key(
        df: pd.DataFrame,
        resin_key_col: str,
        hardener_key_col: str,
        new_col: str = "formulation_key",
) -> pd.DataFrame:
    """基于 resin_key_col + hardener_key_col 构建体系配方键"""
    if resin_key_col not in df.columns or hardener_key_col not in df.columns:
        return df
    new_df = df.copy()
    new_df[new_col] = (
            new_df[resin_key_col].astype(str).fillna("") + "||" + new_df[hardener_key_col].astype(str).fillna("")
    ).replace({"||": np.nan})
    return new_df


def top_value_counts(series: pd.Series, top_n: int = 10) -> pd.Series:
    """安全的 value_counts(top_n)。"""
    try:
        vc = series.value_counts(dropna=False)
        return vc.head(top_n)
    except Exception:
        return pd.Series(dtype=int)


# =============================================================================
# [新增] 增强版 SMILES 修复函数
# =============================================================================

def _advanced_string_cleaning(text: str) -> Optional[str]:
    """
    [增强版] 高级字符串清理，处理更多特殊情况
    """
    if text is None:
        return None
    
    s = str(text).strip()
    
    # 1. 去除各种引号包裹
    quote_pairs = [('"', '"'), ("'", "'"), ('"', '"'), (''', '''), ('「', '」'), ('『', '』')]
    for left, right in quote_pairs:
        if s.startswith(left) and s.endswith(right):
            s = s[len(left):-len(right)].strip()
    
    # 2. 移除常见的前缀标签
    prefixes_to_remove = [
        'SMILES:', 'SMILES：', 'smiles:', 'Smiles:',
        'SELFIES:', 'selfies:',
        'InChI:', 'inchi:',
        'Structure:', 'structure:',
        '分子式:', '结构式:',
    ]
    for prefix in prefixes_to_remove:
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].strip()
    
    # 3. 移除不可见字符和控制字符
    s = ''.join(c for c in s if c.isprintable() or c == '\t')
    s = s.strip()
    
    # 4. 统一全角字符为半角
    fullwidth_map = {
        '（': '(', '）': ')', '【': '[', '】': ']',
        '｛': '{', '｝': '}', '＝': '=', '＃': '#',
        '－': '-', '＋': '+', '．': '.', '；': ';',
        '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
        '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
        'Ｃ': 'C', 'Ｎ': 'N', 'Ｏ': 'O', 'Ｓ': 'S', 'Ｆ': 'F',
        'ｃ': 'c', 'ｎ': 'n', 'ｏ': 'o', 'ｓ': 's',
    }
    for full, half in fullwidth_map.items():
        s = s.replace(full, half)
    
    # 5. 移除常见的注释和备注（括号内的中文说明等）
    s = re.sub(r'\([^()]*[\u4e00-\u9fff]+[^()]*\)', '', s)
    s = re.sub(r'\[[^\[\]]*[\u4e00-\u9fff]+[^\[\]]*\]', '', s)
    s = s.strip()
    
    # 6. 检查无效关键词
    if not s or s.lower() in ['nan', 'none', 'null', 'na', 'n/a', '-', '--', '---', 
                               'unknown', 'n.a.', 'n.a', 'empty', '空', '无', '未知']:
        return None
    
    return s


def _fix_ring_closures(smiles: str) -> Optional[str]:
    """修复环闭合数字不匹配的问题"""
    if not smiles:
        return None
    
    ring_counts = {}
    ring_pattern = re.compile(r'%(\d{2})|(\d)')
    
    for match in ring_pattern.finditer(smiles):
        num = match.group(1) or match.group(2)
        ring_counts[num] = ring_counts.get(num, 0) + 1
    
    unpaired = [num for num, count in ring_counts.items() if count % 2 != 0]
    
    if not unpaired:
        return smiles
    
    result = smiles
    for num in unpaired:
        if len(num) == 1:
            pattern = re.compile(r'(?<![0-9%])' + num + r'(?![0-9])')
        else:
            pattern = re.compile(r'%' + num)
        result = pattern.sub('', result, count=1)
    
    return result if result else None


def _fix_parentheses(smiles: str) -> Optional[str]:
    """修复括号不匹配的问题"""
    if not smiles:
        return None
    
    brackets = {'(': ')', '[': ']', '{': '}'}
    stack = []
    result = list(smiles)
    to_remove = []
    
    for i, c in enumerate(smiles):
        if c in brackets:
            stack.append((c, i))
        elif c in brackets.values():
            if stack and brackets.get(stack[-1][0]) == c:
                stack.pop()
            else:
                to_remove.append(i)
    
    for _, i in stack:
        to_remove.append(i)
    
    for i in sorted(to_remove, reverse=True):
        result.pop(i)
    
    return ''.join(result) if result else None


def aggressive_repair_smiles(smiles: str, keep_largest_frag: bool = True, 
                              preserve_original_on_fail: bool = False) -> Optional[str]:
    """
    [增强版] 激进修复模式：尝试所有可能的修复策略，最大化保留数据
    
    修复策略（按优先级顺序尝试）：
    1. 高级字符串清理
    2. 直接解析
    3. 移除立体化学标记
    4. 移除电荷标记
    5. 处理聚合物占位符 (* -> C)
    6. 移除金属离子
    7. 分割多组分逐个尝试
    8. 修复环闭合数字不匹配
    9. 修复括号不匹配
    10. 宽松模式解析
    11. 组合修复
    12. 保留原始字符串（如果 preserve_original_on_fail=True）
    
    Args:
        smiles: 输入字符串
        keep_largest_frag: 是否只保留最大片段
        preserve_original_on_fail: 如果所有修复都失败，是否返回清理后的原始字符串
        
    Returns:
        修复后的 SMILES，或 None
    """
    if not RDKIT_AVAILABLE:
        return smiles if preserve_original_on_fail else None
    
    # 1. 高级字符串清理
    s = _advanced_string_cleaning(smiles)
    if not s:
        return None
    
    original_cleaned = s
    
    # 2. 直接尝试解析
    canon = canonicalize_smiles(s)
    if canon:
        if keep_largest_frag and '.' in canon:
            return max(canon.split('.'), key=len)
        return canon
    
    # 3. 移除立体化学标记
    s_no_stereo = re.sub(r'[@/\\]', '', s)
    canon = canonicalize_smiles(s_no_stereo)
    if canon:
        if keep_largest_frag and '.' in canon:
            return max(canon.split('.'), key=len)
        return canon
    
    # 4. 移除电荷标记
    s_no_charge = re.sub(r'\[([A-Za-z]+)[+-]\d*\]', r'\1', s)
    s_no_charge = re.sub(r'\[([A-Za-z]+)([+-])\]', r'\1', s_no_charge)
    canon = canonicalize_smiles(s_no_charge)
    if canon:
        if keep_largest_frag and '.' in canon:
            return max(canon.split('.'), key=len)
        return canon
    
    # 5. 处理聚合物占位符
    s_no_star = s.replace('[*]', 'C').replace('*', 'C')
    s_no_star = re.sub(r'\[\s*\*\s*\]', 'C', s_no_star)
    canon = canonicalize_smiles(s_no_star)
    if canon:
        if keep_largest_frag and '.' in canon:
            return max(canon.split('.'), key=len)
        return canon
    
    # 6. 移除常见金属离子片段
    metal_patterns = [
        r'\[Na\+?\]\.?', r'\[K\+?\]\.?', r'\[Li\+?\]\.?',
        r'\[Ca\+?\+?\]\.?', r'\[Mg\+?\+?\]\.?', r'\[Zn\+?\+?\]\.?',
        r'\[Fe\+?\+?\+?\]\.?', r'\[Cu\+?\+?\]\.?',
        r'\[Cl-?\]\.?', r'\[Br-?\]\.?', r'\[I-?\]\.?',
    ]
    s_no_metal = s
    for pattern in metal_patterns:
        s_no_metal = re.sub(pattern, '', s_no_metal, flags=re.IGNORECASE)
    s_no_metal = re.sub(r'\.+', '.', s_no_metal).strip('.')
    if s_no_metal and s_no_metal != s:
        canon = canonicalize_smiles(s_no_metal)
        if canon:
            if keep_largest_frag and '.' in canon:
                return max(canon.split('.'), key=len)
            return canon
    
    # 7. 分割多组分逐个尝试
    separators = ['.', ';', '；', '|', ' + ']
    for sep in separators:
        if sep in s:
            frags = [f.strip() for f in s.split(sep) if f.strip()]
            frags.sort(key=len, reverse=True)
            for frag in frags:
                canon = canonicalize_smiles(frag)
                if canon:
                    return canon
                frag_no_stereo = re.sub(r'[@/\\]', '', frag)
                canon = canonicalize_smiles(frag_no_stereo)
                if canon:
                    return canon
    
    # 8. 修复环闭合
    s_fixed_ring = _fix_ring_closures(s)
    if s_fixed_ring and s_fixed_ring != s:
        canon = canonicalize_smiles(s_fixed_ring)
        if canon:
            if keep_largest_frag and '.' in canon:
                return max(canon.split('.'), key=len)
            return canon
    
    # 9. 修复括号
    s_fixed_paren = _fix_parentheses(s)
    if s_fixed_paren and s_fixed_paren != s:
        canon = canonicalize_smiles(s_fixed_paren)
        if canon:
            if keep_largest_frag and '.' in canon:
                return max(canon.split('.'), key=len)
            return canon
    
    # 10. 宽松模式解析
    result = _canonicalize_smiles_relaxed(s)
    if result:
        if keep_largest_frag and '.' in result:
            return max(result.split('.'), key=len)
        return result
    
    # 11. 组合修复
    s_combined = s
    s_combined = re.sub(r'[@/\\]', '', s_combined)
    s_combined = re.sub(r'\[([A-Za-z]+)[+-]\d*\]', r'\1', s_combined)
    for pattern in metal_patterns:
        s_combined = re.sub(pattern, '', s_combined, flags=re.IGNORECASE)
    s_combined = s_combined.replace('[*]', 'C').replace('*', 'C')
    s_combined = re.sub(r'\.+', '.', s_combined).strip('.')
    
    if s_combined:
        canon = canonicalize_smiles(s_combined)
        if canon:
            if keep_largest_frag and '.' in canon:
                return max(canon.split('.'), key=len)
            return canon
    
    # 12. 保留原始
    if preserve_original_on_fail:
        return original_cleaned
    
    return None


def ultra_repair_smiles(smiles: str, 
                        keep_largest_frag: bool = True,
                        try_all_fragments: bool = True,
                        preserve_original: bool = False) -> Tuple[Optional[str], str]:
    """
    [超级修复模式] 尝试一切可能的修复策略
    
    返回: (修复后的SMILES, 修复状态描述)
    
    修复状态:
    - 'success': 成功解析
    - 'repaired_stereo': 移除立体化学后成功
    - 'repaired_charge': 移除电荷后成功
    - 'repaired_polymer': 处理聚合物标记后成功
    - 'repaired_metal': 移除金属离子后成功
    - 'repaired_fragment': 从片段中提取成功
    - 'repaired_ring': 修复环闭合后成功
    - 'repaired_paren': 修复括号后成功
    - 'repaired_combined': 组合修复后成功
    - 'preserved': 保留原始字符串（未能解析但保留）
    - 'failed': 所有修复均失败
    """
    if not RDKIT_AVAILABLE:
        if preserve_original and smiles:
            return (str(smiles).strip(), 'preserved')
        return (None, 'failed')
    
    s = _advanced_string_cleaning(smiles)
    if not s:
        return (None, 'failed')
    
    original_cleaned = s
    
    # 1. 直接尝试
    canon = canonicalize_smiles(s)
    if canon:
        result = max(canon.split('.'), key=len) if keep_largest_frag and '.' in canon else canon
        return (result, 'success')
    
    # 2. 移除立体化学
    s_no_stereo = re.sub(r'[@/\\]', '', s)
    canon = canonicalize_smiles(s_no_stereo)
    if canon:
        result = max(canon.split('.'), key=len) if keep_largest_frag and '.' in canon else canon
        return (result, 'repaired_stereo')
    
    # 3. 移除电荷
    s_no_charge = re.sub(r'\[([A-Za-z]+)[+-]\d*\]', r'\1', s)
    s_no_charge = re.sub(r'\[([A-Za-z]+)([+-])\]', r'\1', s_no_charge)
    canon = canonicalize_smiles(s_no_charge)
    if canon:
        result = max(canon.split('.'), key=len) if keep_largest_frag and '.' in canon else canon
        return (result, 'repaired_charge')
    
    # 4. 处理聚合物
    s_no_star = s.replace('[*]', 'C').replace('*', 'C')
    canon = canonicalize_smiles(s_no_star)
    if canon:
        result = max(canon.split('.'), key=len) if keep_largest_frag and '.' in canon else canon
        return (result, 'repaired_polymer')
    
    # 5. 移除金属
    metal_patterns = [
        r'\[Na\+?\]\.?', r'\[K\+?\]\.?', r'\[Li\+?\]\.?',
        r'\[Ca\+?\+?\]\.?', r'\[Mg\+?\+?\]\.?',
        r'\[Cl-?\]\.?', r'\[Br-?\]\.?',
    ]
    s_no_metal = s
    for pattern in metal_patterns:
        s_no_metal = re.sub(pattern, '', s_no_metal, flags=re.IGNORECASE)
    s_no_metal = re.sub(r'\.+', '.', s_no_metal).strip('.')
    if s_no_metal:
        canon = canonicalize_smiles(s_no_metal)
        if canon:
            result = max(canon.split('.'), key=len) if keep_largest_frag and '.' in canon else canon
            return (result, 'repaired_metal')
    
    # 6. 尝试片段
    if try_all_fragments:
        for sep in ['.', ';', '；', '|', ' + ']:
            if sep in s:
                frags = [f.strip() for f in s.split(sep) if f.strip()]
                frags.sort(key=len, reverse=True)
                for frag in frags:
                    canon = canonicalize_smiles(frag)
                    if canon:
                        return (canon, 'repaired_fragment')
                    frag_no_stereo = re.sub(r'[@/\\]', '', frag)
                    canon = canonicalize_smiles(frag_no_stereo)
                    if canon:
                        return (canon, 'repaired_fragment')
    
    # 7. 修复环闭合
    s_fixed_ring = _fix_ring_closures(s)
    if s_fixed_ring and s_fixed_ring != s:
        canon = canonicalize_smiles(s_fixed_ring)
        if canon:
            result = max(canon.split('.'), key=len) if keep_largest_frag and '.' in canon else canon
            return (result, 'repaired_ring')
    
    # 8. 修复括号
    s_fixed_paren = _fix_parentheses(s)
    if s_fixed_paren and s_fixed_paren != s:
        canon = canonicalize_smiles(s_fixed_paren)
        if canon:
            result = max(canon.split('.'), key=len) if keep_largest_frag and '.' in canon else canon
            return (result, 'repaired_paren')
    
    # 9. 组合修复
    s_combined = s
    s_combined = re.sub(r'[@/\\]', '', s_combined)
    s_combined = re.sub(r'\[([A-Za-z]+)[+-]\d*\]', r'\1', s_combined)
    for pattern in metal_patterns:
        s_combined = re.sub(pattern, '', s_combined, flags=re.IGNORECASE)
    s_combined = s_combined.replace('[*]', 'C').replace('*', 'C')
    s_combined = re.sub(r'\.+', '.', s_combined).strip('.')
    
    if s_combined:
        canon = canonicalize_smiles(s_combined)
        if canon:
            result = max(canon.split('.'), key=len) if keep_largest_frag and '.' in canon else canon
            return (result, 'repaired_combined')
    
    # 10. 保留原始
    if preserve_original:
        return (original_cleaned, 'preserved')
    
    return (None, 'failed')


# =============================================================================
# Transformer 纠错集成
# =============================================================================

# 全局流水线实例（延迟初始化）
_CORRECTION_PIPELINE = None


def get_correction_pipeline(model_path: Optional[str] = None, 
                           force_reinit: bool = False):
    """
    获取SMILES纠错流水线单例
    
    Args:
        model_path: 预训练模型路径（可选）
        force_reinit: 是否强制重新初始化
    
    Returns:
        SMILESCorrectionPipeline实例
    """
    global _CORRECTION_PIPELINE
    
    if _CORRECTION_PIPELINE is None or force_reinit:
        try:
            from .smiles_transformer_corrector import SMILESCorrectionPipeline
            _CORRECTION_PIPELINE = SMILESCorrectionPipeline()
            
            if model_path:
                _CORRECTION_PIPELINE.load_model(model_path)
        except ImportError as e:
            print(f"警告: 无法导入Transformer纠错模块: {e}")
            _CORRECTION_PIPELINE = None
    
    return _CORRECTION_PIPELINE


def transformer_repair_smiles(smiles: str,
                             model_path: Optional[str] = None,
                             beam_size: int = 5,
                             use_rules_fallback: bool = True,
                             keep_largest_frag: bool = True) -> Optional[str]:
    """
    使用Transformer模型修复SMILES
    
    结合深度学习模型和规则方法的完整修复流程。
    
    Args:
        smiles: 输入SMILES字符串
        model_path: 预训练模型路径（可选）
        beam_size: Beam search大小
        use_rules_fallback: 如果DL失败，是否使用规则方法
        keep_largest_frag: 是否只保留最大片段
    
    Returns:
        修复后的SMILES，失败返回None
    """
    pipeline = get_correction_pipeline(model_path)
    
    if pipeline is not None:
        result = pipeline.correct(
            smiles,
            use_dl=True,
            use_rules=use_rules_fallback,
            beam_size=beam_size
        )
        return result
    
    # 回退到规则方法
    if use_rules_fallback:
        return aggressive_repair_smiles(smiles, keep_largest_frag=keep_largest_frag)
    
    return None


def batch_transformer_repair(smiles_list: List[str],
                            model_path: Optional[str] = None,
                            beam_size: int = 5,
                            use_rules_fallback: bool = True,
                            show_progress: bool = True) -> List[Optional[str]]:
    """
    批量使用Transformer模型修复SMILES
    
    Args:
        smiles_list: SMILES列表
        model_path: 预训练模型路径（可选）
        beam_size: Beam search大小
        use_rules_fallback: 如果DL失败，是否使用规则方法
        show_progress: 是否显示进度条
    
    Returns:
        修复结果列表
    """
    pipeline = get_correction_pipeline(model_path)
    
    if pipeline is not None:
        return pipeline.correct_batch(
            smiles_list,
            use_dl=True,
            use_rules=use_rules_fallback,
            beam_size=beam_size,
            show_progress=show_progress
        )
    
    # 回退到规则方法
    results = []
    iterator = smiles_list
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(smiles_list, desc='SMILES修复')
        except ImportError:
            pass
    
    for smiles in iterator:
        result = aggressive_repair_smiles(smiles) if use_rules_fallback else None
        results.append(result)
    
    return results


def hybrid_repair_smiles(smiles: str,
                        use_transformer: bool = True,
                        use_aggressive: bool = True,
                        use_ultra: bool = True,
                        model_path: Optional[str] = None,
                        keep_largest_frag: bool = True) -> Tuple[Optional[str], str]:
    """
    混合修复策略：按优先级尝试所有可用方法
    
    修复优先级：
    1. 直接RDKit解析
    2. Transformer模型纠错
    3. 激进修复（aggressive_repair_smiles）
    4. 超级修复（ultra_repair_smiles）
    
    Args:
        smiles: 输入SMILES字符串
        use_transformer: 是否使用Transformer模型
        use_aggressive: 是否使用激进修复
        use_ultra: 是否使用超级修复
        model_path: Transformer模型路径
        keep_largest_frag: 是否只保留最大片段
    
    Returns:
        (修复后的SMILES, 使用的方法名称)
    """
    if not RDKIT_AVAILABLE:
        return (None, 'failed_no_rdkit')
    
    # 基础清理
    s = clean_smiles_raw_string(smiles)
    if not s:
        return (None, 'failed_empty')
    
    # 1. 直接解析
    canon = canonicalize_smiles(s)
    if canon:
        if keep_largest_frag and '.' in canon:
            return (max(canon.split('.'), key=len), 'direct')
        return (canon, 'direct')
    
    # 2. Transformer纠错
    if use_transformer:
        pipeline = get_correction_pipeline(model_path)
        if pipeline is not None:
            result = pipeline.correct(s, use_dl=True, use_rules=False)
            if result:
                return (result, 'transformer')
    
    # 3. 激进修复
    if use_aggressive:
        result = aggressive_repair_smiles(s, keep_largest_frag=keep_largest_frag)
        if result:
            return (result, 'aggressive')
    
    # 4. 超级修复
    if use_ultra:
        result, status = ultra_repair_smiles(s, keep_largest_frag=keep_largest_frag)
        if result and status != 'failed':
            return (result, f'ultra_{status}')
    
    return (None, 'failed')


def clean_smiles_column_with_transformer(
        df: 'pd.DataFrame',
        smiles_column: str,
        output_column: str = 'cleaned_smiles',
        status_column: Optional[str] = 'clean_status',
        method_column: Optional[str] = 'clean_method',
        model_path: Optional[str] = None,
        use_transformer: bool = True,
        use_rules: bool = True,
        inplace: bool = False
) -> 'pd.DataFrame':
    """
    使用Transformer模型清洗DataFrame中的SMILES列
    
    Args:
        df: 输入DataFrame
        smiles_column: SMILES列名
        output_column: 输出列名
        status_column: 状态列名（可选）
        method_column: 方法列名（可选）
        model_path: Transformer模型路径
        use_transformer: 是否使用Transformer模型
        use_rules: 是否使用规则方法
        inplace: 是否原地修改
    
    Returns:
        处理后的DataFrame
    """
    import pandas as pd
    
    if not inplace:
        df = df.copy()
    
    pipeline = get_correction_pipeline(model_path) if use_transformer else None
    
    cleaned = []
    statuses = []
    methods = []
    
    for smiles in df[smiles_column]:
        if pipeline is not None:
            result = pipeline.correct(
                smiles, 
                use_dl=use_transformer, 
                use_rules=use_rules,
                return_details=True
            )
            cleaned.append(result.corrected)
            statuses.append(result.status)
            methods.append(result.method)
        else:
            # 回退到规则方法
            result, status = hybrid_repair_smiles(
                smiles, 
                use_transformer=False,
                use_aggressive=use_rules,
                use_ultra=use_rules
            )
            cleaned.append(result)
            statuses.append('valid' if result else 'failed')
            methods.append(status.split('_')[0] if '_' in status else status)
    
    df[output_column] = cleaned
    
    if status_column:
        df[status_column] = statuses
    
    if method_column:
        df[method_column] = methods
    
    return df


def train_smiles_corrector(valid_smiles: List[str],
                          model_save_path: str,
                          val_ratio: float = 0.1,
                          augment_factor: int = 5,
                          **training_kwargs) -> Dict:
    """
    训练SMILES纠错模型
    
    从有效SMILES列表训练Transformer纠错模型。
    
    Args:
        valid_smiles: 有效SMILES列表
        model_save_path: 模型保存路径
        val_ratio: 验证集比例
        augment_factor: 数据增强倍数
        **training_kwargs: 额外的训练参数
            - batch_size: 批大小（默认64，GPU推荐128）
            - device: 训练设备 ('auto', 'cuda', 'cpu')
            - max_samples: 最大训练样本数（None=全部）
            - max_epochs: 最大训练轮数
            - fast_mode: 快速训练模式（使用更小的模型）
            - learning_rate: 学习率
            - d_model: 模型维度
            - n_heads: 注意力头数
            - n_encoder_layers: 编码器层数
            - n_decoder_layers: 解码器层数
    
    Returns:
        训练历史字典
    """
    try:
        from .smiles_transformer_corrector import (
            SMILESCorrectionPipeline,
            TransformerConfig,
            TrainingConfig
        )
    except ImportError as e:
        raise ImportError(f"无法导入Transformer模块: {e}")
    
    # 限制样本数
    max_samples = training_kwargs.get('max_samples')
    if max_samples and len(valid_smiles) > max_samples:
        import random
        valid_smiles = random.sample(valid_smiles, max_samples)
        print(f"📊 限制训练样本数: {max_samples}")
    
    # 设备配置
    device = training_kwargs.get('device', 'auto')
    
    # 快速模式配置
    fast_mode = training_kwargs.get('fast_mode', False)
    
    # 配置 - 根据快速模式调整模型大小
    batch_size = training_kwargs.get('batch_size', 64)
    
    if fast_mode:
        # 快速模式：更小的模型，更快的训练
        print("⚡ 使用快速训练模式（小模型）")
        transformer_config = TransformerConfig(
            d_model=training_kwargs.get('d_model', 128),  # 更小
            n_heads=training_kwargs.get('n_heads', 4),     # 更少
            n_encoder_layers=training_kwargs.get('n_encoder_layers', 2),  # 更少
            n_decoder_layers=training_kwargs.get('n_decoder_layers', 2),  # 更少
            d_ff=training_kwargs.get('d_ff', 512),         # 更小
            dropout=training_kwargs.get('dropout', 0.1)
        )
    else:
        # 标准模式：完整模型
        print("🎯 使用标准训练模式")
        transformer_config = TransformerConfig(
            d_model=training_kwargs.get('d_model', 256),
            n_heads=training_kwargs.get('n_heads', 8),
            n_encoder_layers=training_kwargs.get('n_encoder_layers', 4),
            n_decoder_layers=training_kwargs.get('n_decoder_layers', 4),
            d_ff=training_kwargs.get('d_ff', 1024),
            dropout=training_kwargs.get('dropout', 0.1)
        )
    
    training_config = TrainingConfig(
        batch_size=batch_size,
        learning_rate=training_kwargs.get('learning_rate', 1e-4),
        max_epochs=training_kwargs.get('max_epochs', 100),
        patience=training_kwargs.get('patience', 10),
        noise_prob=training_kwargs.get('noise_prob', 0.15),
        device=device
    )
    
    # 创建流水线
    pipeline = SMILESCorrectionPipeline(
        transformer_config=transformer_config,
        training_config=training_config
    )
    
    # 训练
    history = pipeline.train_from_valid_smiles(
        valid_smiles,
        val_ratio=val_ratio,
        augment_factor=augment_factor,
        save_path=model_save_path
    )
    
    # 更新全局实例
    global _CORRECTION_PIPELINE
    _CORRECTION_PIPELINE = pipeline
    
    return history


# 导出扩展API
__all_transformer__ = [
    'get_correction_pipeline',
    'transformer_repair_smiles',
    'batch_transformer_repair',
    'hybrid_repair_smiles',
    'clean_smiles_column_with_transformer',
    'train_smiles_corrector',
]
