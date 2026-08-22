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

try:
    from rdkit import Chem
    from rdkit.Chem import rdChemReactions
    RDKIT_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    rdChemReactions = None
    RDKIT_AVAILABLE = False


ALLOWED_ELEMENTS = {1, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}


@dataclass
class DesignConfig:
    """Configuration shared by rule-based and model-guided design stages."""

    random_state: int = 0
    max_scaffolds: int = 128
    max_variants_per_scaffold: int = 32
    max_products_per_template: int = 32
    keep_parents: bool = True
    enabled_templates: list[str] = field(default_factory=list)
    search_depth: int = 1
    beam_width: int = 8
    exploration_ratio: float = 0.2


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


@dataclass(frozen=True)
class ReactionTemplate:
    template_id: str
    version: int
    roles: tuple[str, ...]
    reaction_smarts: str
    required_site: str
    risk_level: str = "low"
    max_products: int = 32


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    role_valid: bool
    canonical_smiles: str | None = None
    reasons: tuple[str, ...] = ()


class ReactionTemplateRegistry:
    """Small, versioned registry for the built-in molecule edit templates."""

    _templates = {
        "aryl_methyl_substitution": ReactionTemplate(
            "aryl_methyl_substitution", 1, ("resin", "hardener"),
            "[cH:1]>>[c:1]-[CH3]", "aromatic_hydrogen",
        ),
        "hydroxyl_glycidyl_ether": ReactionTemplate(
            "hydroxyl_glycidyl_ether", 1, ("resin",),
            "[O;H1:1]>>[O:1]CC1CO1", "hydroxyl",
        ),
        "amine_alkylation": ReactionTemplate(
            "amine_alkylation", 1, ("hardener",),
            "[N;H1,H2:1]>>[N:1]C", "amine_hydrogen",
        ),
        "ether_chain_scan": ReactionTemplate(
            "ether_chain_scan", 1, ("resin", "hardener"),
            "[CH3;!R:1]>>[CH2:1]C", "chain_terminal",
        ),
    }

    @classmethod
    def get(cls, template_id: str) -> ReactionTemplate | None:
        return cls._templates.get(str(template_id))

    @classmethod
    def all(cls) -> tuple[ReactionTemplate, ...]:
        return tuple(cls._templates.values())


def _load_mol(smiles: str):
    if not RDKIT_AVAILABLE or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def validate_product(smiles: str, role: str = "neutral") -> ValidationReport:
    """Validate a generated product before feature extraction/model scoring."""
    if not RDKIT_AVAILABLE:
        return ValidationReport(False, False, reasons=("rdkit_unavailable",))
    mol = _load_mol(smiles)
    if mol is None:
        return ValidationReport(False, False, reasons=("invalid_smiles",))
    reasons: list[str] = []
    if "." in str(smiles):
        reasons.append("disconnected_product")
    if any(atom.GetAtomicNum() not in ALLOWED_ELEMENTS for atom in mol.GetAtoms()):
        reasons.append("element_not_allowed")
    try:
        canonical = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        canonical = None
        reasons.append("canonicalization_failed")
    role_key = str(role or "neutral").lower()
    role_valid = role_key in {"neutral", "any", "resin", "hardener"}
    if not role_valid:
        reasons.append("unknown_role")
    return ValidationReport(
        ok=not reasons and role_valid,
        role_valid=role_valid,
        canonical_smiles=canonical,
        reasons=tuple(reasons),
    )


def _append_atom(mol, atomic_num: int = 6) -> int:
    atom = Chem.Atom(int(atomic_num))
    return mol.AddAtom(atom)


def _connect_single(mol, atom_idx: int, atomic_num: int = 6) -> int:
    new_idx = _append_atom(mol, atomic_num)
    mol.AddBond(int(atom_idx), int(new_idx), Chem.BondType.SINGLE)
    return new_idx


def _product_from_rw(mol, parent: str, role: str, template: ReactionTemplate, trace: dict[str, Any]):
    try:
        product_mol = mol.GetMol()
        Chem.SanitizeMol(product_mol)
        product_smiles = Chem.MolToSmiles(product_mol, canonical=True)
    except Exception:
        return None
    report = validate_product(product_smiles, role=role)
    if not report.ok or not report.canonical_smiles:
        return None
    return DesignProduct(
        parent_smiles=parent,
        product_smiles=report.canonical_smiles,
        role=role,
        design_method="reaction_template",
        template_id=template.template_id,
        edit_trace=[trace],
        design_depth=1,
        chemical_validity=True,
    )


def apply_design_template(smiles: str, template_id: str, *, role: str) -> list[DesignProduct]:
    """Apply one built-in template using explicit RDKit atom/bond edits."""
    template = ReactionTemplateRegistry.get(template_id)
    if template is None or str(role) not in template.roles:
        return []
    parent = _load_mol(smiles)
    if parent is None:
        return []
    parent_canonical = Chem.MolToSmiles(parent, canonical=True)
    products: list[DesignProduct] = []
    if template_id == "aryl_methyl_substitution":
        site_indices = [
            atom.GetIdx() for atom in parent.GetAtoms()
            if atom.GetIsAromatic() and atom.GetTotalNumHs() > 0
        ]
        for site_idx in site_indices[: template.max_products]:
            rw = Chem.RWMol(parent)
            new_idx = _connect_single(rw, site_idx, 6)
            product = _product_from_rw(
                rw, parent_canonical, role, template,
                {"site_atom": int(site_idx), "new_atom": int(new_idx), "bond": "SINGLE"},
            )
            if product is not None:
                products.append(product)
    elif template_id == "hydroxyl_glycidyl_ether":
        site_indices = [
            atom.GetIdx() for atom in parent.GetAtoms()
            if atom.GetAtomicNum() == 8 and atom.GetTotalNumHs() > 0
        ]
        for site_idx in site_indices[: template.max_products]:
            rw = Chem.RWMol(parent)
            first = _connect_single(rw, site_idx, 6)
            carbon2 = _connect_single(rw, first, 6)
            oxygen = _append_atom(rw, 8)
            rw.AddBond(carbon2, oxygen, Chem.BondType.SINGLE)
            rw.AddBond(oxygen, first, Chem.BondType.SINGLE)
            product = _product_from_rw(
                rw, parent_canonical, role, template,
                {"site_atom": int(site_idx), "new_atoms": [int(first), int(carbon2), int(oxygen)], "bond": "SINGLE"},
            )
            if product is not None:
                products.append(product)
    elif template_id == "amine_alkylation":
        site_indices = [
            atom.GetIdx() for atom in parent.GetAtoms()
            if atom.GetAtomicNum() == 7 and atom.GetTotalNumHs() > 0
        ]
        for site_idx in site_indices[: template.max_products]:
            rw = Chem.RWMol(parent)
            new_idx = _connect_single(rw, site_idx, 6)
            product = _product_from_rw(
                rw, parent_canonical, role, template,
                {"site_atom": int(site_idx), "new_atom": int(new_idx), "bond": "SINGLE"},
            )
            if product is not None:
                products.append(product)
    elif template_id == "ether_chain_scan":
        site_indices = [
            atom.GetIdx() for atom in parent.GetAtoms()
            if atom.GetAtomicNum() == 6 and not atom.IsInRing() and atom.GetDegree() == 1
        ]
        for site_idx in site_indices[: template.max_products]:
            rw = Chem.RWMol(parent)
            new_idx = _connect_single(rw, site_idx, 6)
            product = _product_from_rw(
                rw, parent_canonical, role, template,
                {"site_atom": int(site_idx), "new_atom": int(new_idx), "bond": "SINGLE", "scan": "C1-C6"},
            )
            if product is not None:
                products.append(product)
    return products


def _count_pattern(mol, smarts: str) -> int:
    if mol is None:
        return 0
    try:
        pattern = Chem.MolFromSmarts(smarts)
        return len(mol.GetSubstructMatches(pattern)) if pattern is not None else 0
    except Exception:
        return 0


def _role_specific_valid(smiles: str, role: str) -> bool:
    """Apply conservative chemistry-role gates after a product is connected."""
    mol = _load_mol(smiles)
    role_key = str(role or "neutral").lower()
    if mol is None:
        return False
    if role_key == "resin":
        return _count_pattern(mol, "[O;r3]1[C;r3][C;r3]1") > 0
    if role_key == "hardener":
        active = sum(int(atom.GetAtomicNum() == 7 and atom.GetTotalNumHs() > 0) for atom in mol.GetAtoms())
        active += _count_pattern(mol, "[SX2H]")
        active += _count_pattern(mol, "[cX3][OX2H]")
        active += _count_pattern(mol, "[CX3](=O)O[CX3](=O)")
        active += _count_pattern(mol, "n1cc[nH]c1")
        return active > 0
    return True


def generate_rule_based_variants(
    scaffolds: Sequence[Scaffold],
    config: DesignConfig,
) -> list[DesignProduct]:
    """Generate deterministic A/B variants under role and quota constraints."""
    template_ids = list(config.enabled_templates)
    if not template_ids:
        template_ids = [template.template_id for template in ReactionTemplateRegistry.all()]
    products: list[DesignProduct] = []
    for scaffold in list(scaffolds)[: max(0, int(config.max_scaffolds))]:
        parent_report = validate_product(scaffold.smiles, role=scaffold.role)
        if not parent_report.ok or not _role_specific_valid(scaffold.smiles, scaffold.role):
            continue
        if config.keep_parents:
            products.append(
                DesignProduct(
                    parent_smiles=parent_report.canonical_smiles or scaffold.smiles,
                    product_smiles=parent_report.canonical_smiles or scaffold.smiles,
                    role=scaffold.role,
                    design_method="parent",
                    template_id="",
                    design_depth=0,
                    chemical_validity=True,
                )
            )
        generated = 0
        seen_products = {products[-1].product_smiles} if products and products[-1].parent_smiles == scaffold.smiles else set()
        for template_id in template_ids:
            if generated >= int(config.max_variants_per_scaffold):
                break
            for product in apply_design_template(scaffold.smiles, template_id, role=scaffold.role):
                if not _role_specific_valid(product.product_smiles, scaffold.role):
                    continue
                if product.product_smiles in seen_products:
                    continue
                seen_products.add(product.product_smiles)
                products.append(product)
                generated += 1
                if generated >= int(config.max_variants_per_scaffold):
                    break
    return products


def _score_sort_key(product: DesignProduct) -> tuple[float, str, str]:
    score = product.model_score
    numeric = float(score) if score is not None and math.isfinite(float(score)) else float("-inf")
    return (-numeric, str(product.product_smiles), str(product.template_id))


def score_design_products(products: Sequence[DesignProduct], scorer=None) -> list[DesignProduct]:
    """Attach model scores supplied by the screening layer and sort stably."""
    items = list(products)
    if not items:
        return []
    if scorer is not None:
        scores = list(scorer(items))
        if len(scores) != len(items):
            raise ValueError("设计评分器返回的分数数量与候选数量不一致")
        for product, score in zip(items, scores):
            try:
                product.model_score = float(score)
                product.score_source = "model"
            except (TypeError, ValueError):
                product.model_score = None
                product.score_source = "invalid_score"
    return sorted(items, key=_score_sort_key)


class ModelGuidedGraphSearch:
    """Deterministic beam search over chemically validated template edits."""

    def __init__(self, config: SearchConfig, scorer=None, template_ids: Sequence[str] | None = None):
        self.config = config
        self.scorer = scorer
        self.template_ids = list(template_ids or [template.template_id for template in ReactionTemplateRegistry.all()])

    def search(self, seeds: Sequence[DesignProduct]) -> list[DesignProduct]:
        beam = score_design_products(list(seeds), self.scorer)
        if not beam:
            return []
        beam = beam[: max(1, int(self.config.beam_width))]
        for _depth in range(max(0, int(self.config.depth))):
            expanded: list[DesignProduct] = list(beam)
            for parent in beam:
                for template_id in self.template_ids:
                    children = apply_design_template(parent.product_smiles, template_id, role=parent.role)
                    for child in children[: max(1, int(self.config.candidates_per_parent))]:
                        child.parent_smiles = parent.product_smiles
                        child.design_depth = parent.design_depth + 1
                        child.edit_trace = list(parent.edit_trace) + list(child.edit_trace)
                        expanded.append(child)
            deduped: dict[tuple[str, str], DesignProduct] = {}
            for item in expanded:
                key = (str(item.product_smiles), str(item.role))
                deduped.setdefault(key, item)
            ranked = score_design_products(list(deduped.values()), self.scorer)
            beam = ranked[: max(1, int(self.config.beam_width))]
            if not beam:
                break
        return score_design_products(beam, self.scorer)[: max(1, int(self.config.max_products))]


def search_design_space(
    seeds: Sequence[DesignProduct],
    config: SearchConfig,
    scorer=None,
) -> list[DesignProduct]:
    return ModelGuidedGraphSearch(config, scorer=scorer).search(seeds)


MoleculeDesignResult = DesignResult


def design_molecules(
    scaffolds: Sequence[Scaffold],
    config: DesignConfig,
    *,
    model=None,
    pipeline=None,
    feature_cols=None,
    scorer=None,
) -> DesignResult:
    """Orchestrate rule edits and optional model-guided graph search.

    The screening page owns feature extraction because the saved workflow may
    be multi-step. It passes a scorer callback here after that workflow has
    produced a contract-validated feature matrix.
    """
    del pipeline, feature_cols
    scaffold_list = list(scaffolds or [])[: max(0, int(config.max_scaffolds))]
    result = DesignResult(config=config, design_hash=compute_design_hash({"config": config, "scaffolds": scaffold_list}))
    if not scaffold_list:
        result.prediction_block_reason = "没有可用于分子设计的有效骨架"
        result.stage_counts = {"scaffolds": 0, "template_products": 0, "valid_products": 0, "scored_products": 0}
        return result
    variants = generate_rule_based_variants(scaffold_list, config)
    result.stage_counts["scaffolds"] = len(scaffold_list)
    result.stage_counts["template_products"] = len(variants)
    if not variants:
        result.prediction_block_reason = "骨架没有生成任何满足角色规则的结构变体"
        result.stage_counts.update({"valid_products": 0, "scored_products": 0})
        return result
    if int(config.max_variants_per_scaffold) > 0:
        search_config = SearchConfig(
            depth=max(0, int(config.search_depth)),
            beam_width=max(1, min(int(config.beam_width), int(config.max_variants_per_scaffold), 32)),
            max_products=max(1, int(config.max_variants_per_scaffold) * max(1, len(scaffold_list))),
            random_state=int(config.random_state),
            exploration_ratio=float(config.exploration_ratio),
        )
        variants = search_design_space(variants, search_config, scorer=scorer)
    result.products = variants
    result.stage_counts["valid_products"] = len(variants)
    result.stage_counts["scored_products"] = sum(item.model_score is not None for item in variants)
    if scorer is None and model is None:
        result.prediction_block_reason = "未提供模型或特征评分器，不能进入预测阶段"
        result.can_predict = False
    else:
        result.can_predict = bool(variants)
    return result


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
    "ALLOWED_ELEMENTS",
    "ReactionTemplate",
    "ReactionTemplateRegistry",
    "ValidationReport",
    "validate_product",
    "apply_design_template",
    "generate_rule_based_variants",
    "score_design_products",
    "ModelGuidedGraphSearch",
    "search_design_space",
    "MoleculeDesignResult",
    "design_molecules",
]
