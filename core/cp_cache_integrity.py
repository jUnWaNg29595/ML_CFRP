# -*- coding: utf-8 -*-
"""Stable, fail-closed input fingerprint for CFRP component-physics (cp) caches.

Fingerprinting is NOT a chemistry model. Its sole contract is:
A precomputed cp may be trusted only if it was built from the same raw
formulation facts using the same component-physics assumptions.

A fingerprint is computed over *source inputs*, never over cp-generated
outputs. Both values and the set of provided columns are tracked. A source
field added/deleted/changed invalidates the cache, including changes that
leave total molecular counts invariant (EEW, AHEW, functionality, SMILES).

The hash is deterministic across column reordering, CSV numeric round trips,
and Python runs, and does not incorporate mutable DataFrame index labels.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import re
from typing import Any

import numpy as np
import pandas as pd

VERSION = "cp-input-v1"
CACHE_HASH_COL = "cp_cache_input_hash"
CACHE_MODE_COL = "cp_cache_assume_stoich"
CACHE_MAX_COL = "cp_cache_max_components"

# Any source-side field that might affect molecular properties, formulation
# equivalents, r provenance, or composition coverage is a dependency.
_ROLE_PATTERN = re.compile(r"^(resin|epoxy|curing_agent|curingagent|hardener|curer)(?:_|$)")
# These are output-only fields written by compute_component_physics; they
# must not create false cache invalidations if already present in the table.
_GENERATED_SUFFIXES = frozenset((
    "mw_resolved", "ew_resolved", "mw_source", "ew_source",
    "f_source", "mw_trust", "structure_used", "mechanism",
    "bigsmiles_proxy", "multi_fragment", "struct_func_conflict",
    "has_structure",
))
# f_stoich and f_network deliberately remain inputs: an upstream program
# may supply/modify those columns, and crosslink_physics can consume them.

_SOURCE_EXACT = frozenset((
    "r_value", "r_actual", "r_source", "xl_r_value", "xl_r_source",
    "rxn_r_actual", "rxn_r_effective", "rxn_r_assumed", "rxn_r_source",
    "cp_r_value", "cp_r_source", "cp_r_partial", "cp_r_unknown",
    "cp_r_from_mass", "cp_composition_incomplete", "cp_unknown_comp_count",
    "curing_mechanism", "stoichiometric_ratio_r_cleaned",
    "crosslink_stoichiometry_r", "resin_total_phr",
))


def _is_source_col(column: Any) -> bool:
    if not isinstance(column, str):
        return False
    if column in (CACHE_HASH_COL, CACHE_MODE_COL, CACHE_MAX_COL):
        return False
    if column in _SOURCE_EXACT or column.startswith("formulation_"):
        return True
    if not _ROLE_PATTERN.match(column):
        return False
    # Exclude computed component outputs; same suffix could appear in a raw
    # table, but a computed value should not be used as new evidence.
    return not any(column.endswith("_" + s) for s in _GENERATED_SUFFIXES)


def _scalar(value: Any, *, column: str) -> str | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (float, int, bool)):
        if isinstance(value, bool):
            return "true" if value else "false"
        try:
            if not math.isfinite(float(value)):
                return None
        except (TypeError, ValueError, OverflowError):
            return str(value)
        return _numeric(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in ("", "nan", "none", "null", "<na>"):
            return None
        # Structural strings are never treated as numbers. For numeric-looking
        # source fields, normalize 8.830 and 8.83 to the same representation.
        if not any(p in column for p in ("structure", "smiles", "name", "source", "mechanism")):
            try:
                return _numeric(stripped)
            except (InvalidOperation, ValueError):
                pass
        return stripped
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _numeric(value: Any) -> str | None:
    try:
        dec = Decimal(str(value))
        if not dec.is_finite():
            return None
        if dec == 0:
            return "0"
        return str(dec.normalize())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("not a numeric value") from exc


def formulation_input_hashes(
    df: pd.DataFrame,
    *,
    max_components: int = 3,
    assume_stoichiometric_if_missing: bool = True,
) -> pd.Series:
    """Compute one stable SHA-256 per input row. Index labels are ignored.

    If duplicate *column names* exist, fail explicitly: there is no safe way
    to identify which physical input column the upstream functions used.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not df.columns.is_unique:
        raise ValueError("Cannot fingerprint cp inputs with duplicate column names")
    cols = sorted(c for c in df.columns if _is_source_col(c))
    preamble = [VERSION, int(max_components), bool(assume_stoichiometric_if_missing), cols]
    result: list[str] = []
    # Only dependency columns are traversed; never RDKit/SMILES re-parsing.
    for row in df[cols].itertuples(index=False, name=None):
        canonical = [preamble, [_scalar(v, column=c) for c, v in zip(cols, row)]]
        blob = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        result.append(hashlib.sha256(blob.encode("utf-8")).hexdigest())
    return pd.Series(result, index=df.index, dtype="object", name=CACHE_HASH_COL)


def stamp_cp_cache(
    cp: pd.DataFrame,
    raw_df: pd.DataFrame,
    *,
    max_components: int = 3,
    assume_stoichiometric_if_missing: bool = True,
) -> pd.DataFrame:
    """Attach cache evidence without changing numeric physical features."""
    if len(cp) != len(raw_df):
        raise ValueError("cp and raw_df row counts differ")
    cp = cp.copy()
    cp[CACHE_HASH_COL] = formulation_input_hashes(
        raw_df, max_components=max_components,
        assume_stoichiometric_if_missing=assume_stoichiometric_if_missing,
    ).to_numpy()
    cp[CACHE_MODE_COL] = bool(assume_stoichiometric_if_missing)
    cp[CACHE_MAX_COL] = int(max_components)
    return cp


def invalid_cp_cache_mask(
    raw_df: pd.DataFrame,
    cp: pd.DataFrame,
    *,
    max_components: int = 3,
    assume_stoichiometric_if_missing: bool = True,
) -> np.ndarray:
    """Return per-position invalidity. Legacy/unverifiable cache => invalid.

    No comparisons of r, MW, or amount with arbitrary numeric thresholds.
    All numeric fields are included in the input fingerprint, so a small
    quantity change is detected without RDKit re-parsing.
    """
    n = len(raw_df)
    if not isinstance(cp, pd.DataFrame) or len(cp) != n:
        return np.ones(n, dtype=bool)
    # pandas aligns many downstream cp columns by labels, not by position.
    # Equal values with differently labeled indices are not a safe cache.
    if not cp.index.equals(raw_df.index):
        return np.ones(n, dtype=bool)
    required = (CACHE_HASH_COL, CACHE_MODE_COL, CACHE_MAX_COL)
    if any(col not in cp.columns for col in required):
        return np.ones(n, dtype=bool)
    expected = formulation_input_hashes(
        raw_df, max_components=max_components,
        assume_stoichiometric_if_missing=assume_stoichiometric_if_missing,
    ).to_numpy()
    saved = cp[CACHE_HASH_COL].to_numpy()
    # bool('False') is True! Do not cast CSV strings to bool.
    expected_mode = "true" if assume_stoichiometric_if_missing else "false"
    modes = np.array([_scalar(v, column=CACHE_MODE_COL) for v in cp[CACHE_MODE_COL].to_numpy()], dtype=object)
    modes = np.array(["true" if str(v).lower() in ("true", "1") else "false" if str(v).lower() in ("false", "0") else "invalid" for v in modes], dtype=object)
    maximums = pd.to_numeric(cp[CACHE_MAX_COL], errors="coerce").to_numpy()
    return (saved != expected) | (modes != expected_mode) | (maximums != int(max_components))
