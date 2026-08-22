"""Domain types and scaffold extraction for virtual molecule design.

This module intentionally contains no generation or model code yet.  It defines
the stable data contracts shared by the later design, validation, and search
tasks, plus the small amount of input normalization needed to create them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import pandas as pd

from .smiles_utils import normalize_chemical_string


@dataclass
class DesignConfig:
    """Configuration shared by rule-based and model-guided design stages."""

    random_state: int = 0
    max_scaffolds: int = 128
    max_variants_per_scaffold: int = 32
    max_products_per_template: int = 32
    keep_parents: bool = True
    enabled_templates: list[str] = field(default_factory=list)


@dataclass
class SearchConfig:
    """Deterministic constrained-search settings."""

    depth: int = 1
    beam_width: int = 8
    candidates_per_parent: int = 32
    exploration_ratio: float = 0.2
    random_state: int = 0
    max_products: int = 128


@dataclass
class Scaffold:
    """A validated, canonical parent structure and its provenance."""

    smiles: str
    role: str
    source: str = "frame"
    source_index: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_row(self) -> Any:
        """Alias used by consumers that call the preserved index a row."""

        return self.source_index

    @property
    def row_index(self) -> Any:
        """Backward-compatible alias for the original frame index."""

        return self.source_index


@dataclass
class DesignProduct:
    """A generated or retained product with an auditable design trace."""

    parent_smiles: str
    product_smiles: str
    role: str
    design_method: str
    template_id: str
    edit_trace: list[dict[str, Any]] = field(default_factory=list)
    design_depth: int = 0
    chemical_validity: bool = False
    filter_reason: str | None = None
    prediction: float | None = None
    prediction_std: float | None = None
    applicability_score: float | None = None
    synth_score: float | None = None
    model_score: float | None = None
    score_source: str | None = None


@dataclass
class DesignResult:
    """Container for design products, failures, and the configuration used."""

    products: list[DesignProduct] = field(default_factory=list)
    failures: list[dict[str, Any] | str] = field(default_factory=list)
    config: DesignConfig | None = None
    design_hash: str | None = None
    prediction_block_reason: str | None = None
    can_predict: bool = False
    stage_counts: dict[str, int] = field(default_factory=dict)


def _json_safe(value: Any) -> Any:
    """Convert dataclasses and common dataframe scalar values to JSON values."""

    if is_dataclass(value):
        return _json_safe(asdict(value))
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized_items = [_json_safe(item) for item in value]
        return sorted(
            normalized_items,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )

    # numpy and pandas scalar objects expose ``item`` while remaining optional
    # dependencies of this lightweight contract.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def compute_design_hash(value: Any) -> str:
    """Return a stable SHA-256 hash of a JSON-safe design value.

    Dataclass fields and mapping keys are serialized with sorted keys so the
    hash is independent of object construction order.
    """

    payload = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ScaffoldMiner:
    """Extract canonical, deduplicated scaffolds from tabular data."""

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        role: str,
        smiles_columns: str | Sequence[str],
        max_scaffolds: int,
        random_state: int | None = None,
    ) -> list[Scaffold]:
        """Normalize SMILES values while preserving row order and provenance.

        ``random_state`` is accepted as part of the public contract.  Mining is
        deliberately stable and does not shuffle rows, so repeated calls with
        the same frame produce the same order regardless of that seed.
        """

        if frame is None or not hasattr(frame, "iterrows"):
            raise TypeError("frame must be a pandas-like DataFrame")
        if max_scaffolds <= 0:
            return []
        columns = [smiles_columns] if isinstance(smiles_columns, str) else list(smiles_columns)
        if not columns:
            return []
        available_columns = [column for column in columns if column in frame.columns]
        if not available_columns:
            return []

        del random_state  # Stability is provided by first-seen traversal.
        results: list[Scaffold] = []
        seen: set[str] = set()
        for source_index, row in frame.iterrows():
            for column in available_columns:
                normalized = normalize_chemical_string(row[column])
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                results.append(
                    Scaffold(
                        smiles=normalized,
                        role=str(role),
                        source="frame",
                        source_index=source_index,
                        metadata={"smiles_column": str(column)},
                    )
                )
                if len(results) >= max_scaffolds:
                    return results
        return results


__all__ = [
    "DesignConfig",
    "SearchConfig",
    "Scaffold",
    "DesignProduct",
    "DesignResult",
    "ScaffoldMiner",
    "compute_design_hash",
]
