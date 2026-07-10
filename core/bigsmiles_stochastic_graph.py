from __future__ import annotations

import random
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from rdkit import Chem

    RDKIT_AVAILABLE = True
except Exception:
    Chem = None
    RDKIT_AVAILABLE = False

from .smiles_utils import canonicalize_smiles, detect_chem_string_format


_BIGSMILES_PREFIX_RE = re.compile(r"^BIGSMILES\s*[:：]\s*", flags=re.I)
_WEIGHT_BLOCK_RE = re.compile(r"\|([^|]+)\|")
_EMPTY_CONNECTOR_RE = re.compile(r"\[\s*\]")
_DUMMY_CONNECTOR_RE = re.compile(
    r"\[\s*(?:<|>|\*|\$)(?:[A-Za-z0-9_.:+-]+)?(?:\|[^]]*\|)?\s*\]"
)
_STAR_RE = re.compile(r"(?<!\[)\*(?!\])")
_EMPTY_BRANCH_RE = re.compile(r"\(\)")
_MULTI_DOT_RE = re.compile(r"\.+")
_LEADING_DOT_RE = re.compile(r"^[.;,\s]+")
_TRAILING_DOT_RE = re.compile(r"[.;,\s]+$")
_NUMERIC_RE = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)")


@dataclass
class BigSMILESUnit:
    raw: str
    smiles_with_dummies: str
    proxy_smiles: str
    weight: float = 1.0
    connector_count: int = 0


@dataclass
class BigSMILESBlock:
    index: int
    raw: str
    repeat_units: List[BigSMILESUnit] = field(default_factory=list)
    end_groups: List[BigSMILESUnit] = field(default_factory=list)
    segment_id: int = 0
    segment_weight: float = 1.0


@dataclass
class BigSMILESEdge:
    source: str
    target: str
    edge_type: str
    weight: float = 1.0


@dataclass
class BigSMILESStochasticGraph:
    raw: str
    prefix_fragments: List[str]
    suffix_fragments: List[str]
    free_fragments: List[str]
    blocks: List[BigSMILESBlock]
    edges: List[BigSMILESEdge]

    def summary(self) -> dict:
        return {
            "n_blocks": len(self.blocks),
            "n_repeat_unit_candidates": int(sum(len(b.repeat_units) for b in self.blocks)),
            "n_end_group_candidates": int(sum(len(b.end_groups) for b in self.blocks)),
            "n_prefix_fragments": len(self.prefix_fragments),
            "n_suffix_fragments": len(self.suffix_fragments),
            "n_free_fragments": len(self.free_fragments),
            "n_edges": len(self.edges),
            "n_segments": len({b.segment_id for b in self.blocks}),
        }


def looks_like_bigsmiles(text) -> bool:
    try:
        return detect_chem_string_format(str(text)) == "bigsmiles"
    except Exception:
        return False


def _split_top_level(text: str, separators: str) -> List[str]:
    if text is None:
        return []
    parts: List[str] = []
    buf: List[str] = []
    brace_depth = 0
    bracket_depth = 0
    paren_depth = 0
    for ch in str(text):
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

        if (
            ch in separators
            and brace_depth == 0
            and bracket_depth == 0
            and paren_depth == 0
        ):
            item = "".join(buf).strip()
            if item:
                parts.append(item)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _tokenize_top_level_sequence(text: str) -> List[tuple[str, str]]:
    text = _BIGSMILES_PREFIX_RE.sub("", str(text or "").strip())
    tokens: List[tuple[str, str]] = []
    buf: List[str] = []
    block_buf: List[str] = []
    depth = 0
    for ch in text:
        if ch == "{":
            if depth == 0:
                out = "".join(buf)
                if out.strip():
                    tokens.append(("text", out))
                buf = []
                block_buf = []
            else:
                block_buf.append(ch)
            depth += 1
            continue
        if ch == "}":
            depth = max(0, depth - 1)
            if depth == 0:
                block = "".join(block_buf).strip()
                if block:
                    tokens.append(("block", block))
                block_buf = []
            else:
                block_buf.append(ch)
            continue
        if depth > 0:
            block_buf.append(ch)
        else:
            buf.append(ch)
    tail = "".join(buf)
    if tail.strip():
        tokens.append(("text", tail))
    return tokens


def _parse_weight_value(spec: str) -> Optional[float]:
    spec = str(spec or "").strip()
    if not spec:
        return None
    nums = [float(x) for x in _NUMERIC_RE.findall(spec)]
    if not nums:
        return None
    if len(nums) == 1:
        return nums[0] if nums[0] > 0 else None
    mean_val = sum(nums) / len(nums)
    return mean_val if mean_val > 0 else None


def _extract_weight_directives(text: str) -> List[float]:
    weights: List[float] = []
    for spec in _WEIGHT_BLOCK_RE.findall(str(text or "")):
        value = _parse_weight_value(spec)
        if value is not None:
            weights.append(float(value))
    return weights


def _strip_weight_directives(text: str) -> str:
    return _WEIGHT_BLOCK_RE.sub("", str(text or ""))


def _strip_nonchemical_markers(text: str) -> str:
    s = _BIGSMILES_PREFIX_RE.sub("", str(text or "").strip())
    if not s:
        return ""
    s = _strip_weight_directives(s)
    s = _EMPTY_CONNECTOR_RE.sub("", s)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", "", s)
    return s


def _cleanup_descriptor_string(text: str, keep_dummies: bool) -> str:
    s = _strip_nonchemical_markers(text)
    if not s:
        return ""

    s = re.sub(r"\|[^|]*\|", "", s)

    if keep_dummies:
        s = _DUMMY_CONNECTOR_RE.sub("[*]", s)
        s = _STAR_RE.sub("[*]", s)
    else:
        s = _DUMMY_CONNECTOR_RE.sub("", s)
        s = s.replace("[*]", "")
        s = s.replace("*", "")

    while True:
        s_next = _EMPTY_BRANCH_RE.sub("", s)
        if s_next == s:
            break
        s = s_next

    s = s.replace(",.", ".").replace(".,", ".").replace(",,", ",")
    s = s.replace(";.", ".").replace(".;", ".")
    s = s.replace(",;", ",").replace(";,", ",")
    s = _MULTI_DOT_RE.sub(".", s)
    s = _LEADING_DOT_RE.sub("", s)
    s = _TRAILING_DOT_RE.sub("", s)
    return s


def _make_proxy_smiles(text: str) -> str:
    s = _cleanup_descriptor_string(text, keep_dummies=False)
    if not s:
        return ""
    can = canonicalize_smiles(s)
    return can if can else s


def _extract_inline_weight(text: str) -> tuple[str, float]:
    s = str(text or "").strip()
    if not s:
        return "", 1.0
    weights = _extract_weight_directives(s)
    if weights:
        return _strip_weight_directives(s).strip(), max(weights)
    return s, 1.0


def _make_unit(token: str) -> Optional[BigSMILESUnit]:
    token = str(token or "").strip()
    if not token:
        return None
    token, weight = _extract_inline_weight(token)
    smiles_with_dummies = _cleanup_descriptor_string(token, keep_dummies=True)
    proxy_smiles = _make_proxy_smiles(token)
    if not smiles_with_dummies and not proxy_smiles:
        return None
    connector_count = smiles_with_dummies.count("[*]")
    return BigSMILESUnit(
        raw=token,
        smiles_with_dummies=smiles_with_dummies or proxy_smiles,
        proxy_smiles=proxy_smiles or smiles_with_dummies,
        weight=float(weight if weight > 0 else 1.0),
        connector_count=int(connector_count),
    )


def _tokenize_fragment_group(text: str) -> List[str]:
    parts: List[str] = []
    for token in _split_top_level(text, ",;"):
        token = token.strip()
        if token:
            parts.append(token)
    return parts


def _extract_free_fragments(text: str) -> List[str]:
    stripped = _strip_weight_directives(text)
    stripped = _BIGSMILES_PREFIX_RE.sub("", stripped)
    stripped = _DUMMY_CONNECTOR_RE.sub("", stripped)
    stripped = stripped.replace("{", "").replace("}", "")
    raw_frags = _split_top_level(stripped, ".;")
    fragments: List[str] = []
    for frag in raw_frags:
        proxy = _make_proxy_smiles(frag)
        if proxy:
            fragments.append(proxy)
    return fragments


def _segment_has_boundary(text: str) -> bool:
    stripped = _strip_weight_directives(text)
    if not stripped.strip():
        return False
    cleaned = stripped.strip()
    return "." in cleaned or ";" in cleaned or bool(_extract_free_fragments(cleaned))


def parse_bigsmiles_stochastic_graph(text) -> Optional[BigSMILESStochasticGraph]:
    if not looks_like_bigsmiles(text):
        return None
    raw = str(text or "").strip()
    sequence = _tokenize_top_level_sequence(raw)
    if not sequence or not any(kind == "block" for kind, _ in sequence):
        return None

    blocks: List[BigSMILESBlock] = []
    edges: List[BigSMILESEdge] = []
    free_fragments: List[str] = []
    prefix_fragments: List[str] = []
    suffix_fragments: List[str] = []

    current_segment = 0
    last_block_idx: Optional[int] = None
    seen_block = False

    for pos, (kind, content) in enumerate(sequence):
        if kind == "block":
            sections = _split_top_level(content, ";")
            repeat_tokens = _tokenize_fragment_group(sections[0] if sections else content)
            end_tokens: List[str] = []
            for sec in sections[1:]:
                end_tokens.extend(_tokenize_fragment_group(sec))

            repeat_units = [unit for unit in (_make_unit(tok) for tok in repeat_tokens) if unit is not None]
            end_groups = [unit for unit in (_make_unit(tok) for tok in end_tokens) if unit is not None]
            block = BigSMILESBlock(
                index=len(blocks),
                raw=content,
                repeat_units=repeat_units,
                end_groups=end_groups,
                segment_id=current_segment,
                segment_weight=1.0,
            )
            blocks.append(block)
            block_id = f"block:{block.index}"
            for unit_idx, unit in enumerate(repeat_units):
                edges.append(
                    BigSMILESEdge(
                        source=block_id,
                        target=f"repeat:{block.index}:{unit_idx}",
                        edge_type="repeat_candidate",
                        weight=float(unit.weight),
                    )
                )
            for unit_idx, unit in enumerate(end_groups):
                edges.append(
                    BigSMILESEdge(
                        source=block_id,
                        target=f"end:{block.index}:{unit_idx}",
                        edge_type="end_group_candidate",
                        weight=float(unit.weight),
                    )
                )
            if last_block_idx is not None and blocks[last_block_idx].segment_id != block.segment_id:
                edges.append(
                    BigSMILESEdge(
                        source=f"block:{last_block_idx}",
                        target=block_id,
                        edge_type="segment_boundary",
                        weight=1.0,
                    )
                )
            last_block_idx = block.index
            seen_block = True
            continue

        weights = _extract_weight_directives(content)
        if weights and last_block_idx is not None:
            blocks[last_block_idx].segment_weight = max(weights)

        fragments = _extract_free_fragments(content)
        if fragments:
            free_fragments.extend(fragments)
            if not seen_block:
                prefix_fragments.extend(fragments)
            elif pos == len(sequence) - 1:
                suffix_fragments.extend(fragments)

        if _segment_has_boundary(content) and last_block_idx is not None:
            current_segment += 1
            last_block_idx = None

    return BigSMILESStochasticGraph(
        raw=raw,
        prefix_fragments=prefix_fragments,
        suffix_fragments=suffix_fragments,
        free_fragments=list(OrderedDict((x, None) for x in free_fragments if x).keys()),
        blocks=blocks,
        edges=edges,
    )


def _weighted_choice(units: List[BigSMILESUnit], rng: random.Random) -> Optional[BigSMILESUnit]:
    if not units:
        return None
    weights = [max(float(unit.weight), 1e-6) for unit in units]
    total = sum(weights)
    if total <= 0:
        return rng.choice(units)
    target = rng.random() * total
    acc = 0.0
    for unit, w in zip(units, weights):
        acc += w
        if acc >= target:
            return unit
    return units[-1]


def _weighted_choice_block(blocks: List[BigSMILESBlock], rng: random.Random) -> Optional[BigSMILESBlock]:
    if not blocks:
        return None
    weights = [max(float(block.segment_weight), 1e-6) for block in blocks]
    total = sum(weights)
    if total <= 0:
        return rng.choice(blocks)
    target = rng.random() * total
    acc = 0.0
    for block, w in zip(blocks, weights):
        acc += w
        if acc >= target:
            return block
    return blocks[-1]


def _mol_from_smiles_with_dummies(smiles: str):
    if not RDKIT_AVAILABLE or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return mol
    except Exception:
        pass
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def _first_dummy_idx(mol) -> Optional[int]:
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            return atom.GetIdx()
    return None


def _remove_all_dummies(mol):
    if not RDKIT_AVAILABLE or mol is None:
        return None
    try:
        rw = Chem.RWMol(mol)
        dummy_indices = [
            atom.GetIdx() for atom in rw.GetAtoms() if atom.GetAtomicNum() == 0
        ]
        for idx in sorted(dummy_indices, reverse=True):
            atom = rw.GetAtomWithIdx(idx)
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(neighbors) == 2 and rw.GetBondBetweenAtoms(neighbors[0], neighbors[1]) is None:
                rw.AddBond(neighbors[0], neighbors[1], Chem.rdchem.BondType.SINGLE)
            rw.RemoveAtom(idx)
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        return out
    except Exception:
        return mol


def _connect_mols_via_dummies(left, right):
    if not RDKIT_AVAILABLE or left is None or right is None:
        return None
    try:
        left_dummy = _first_dummy_idx(left)
        right_dummy = _first_dummy_idx(right)
        if left_dummy is None or right_dummy is None:
            return None
        combo = Chem.CombineMols(left, right)
        rw = Chem.RWMol(combo)
        offset = left.GetNumAtoms()
        left_dummy_rw = left_dummy
        right_dummy_rw = right_dummy + offset
        left_atom = rw.GetAtomWithIdx(left_dummy_rw)
        right_atom = rw.GetAtomWithIdx(right_dummy_rw)
        left_neighbors = [n.GetIdx() for n in left_atom.GetNeighbors()]
        right_neighbors = [n.GetIdx() for n in right_atom.GetNeighbors()]
        if len(left_neighbors) != 1 or len(right_neighbors) != 1:
            return None
        left_anchor = left_neighbors[0]
        right_anchor = right_neighbors[0]
        if rw.GetBondBetweenAtoms(left_anchor, right_anchor) is None:
            rw.AddBond(left_anchor, right_anchor, Chem.rdchem.BondType.SINGLE)
        for idx in sorted([left_dummy_rw, right_dummy_rw], reverse=True):
            rw.RemoveAtom(idx)
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        return out
    except Exception:
        return None


def _fallback_join_smiles(units: List[BigSMILESUnit], free_fragments: List[str]) -> Optional[str]:
    parts: List[str] = []
    for unit in units or []:
        proxy = unit.proxy_smiles or _make_proxy_smiles(unit.raw)
        if proxy:
            parts.append(proxy)
    for frag in free_fragments or []:
        proxy = _make_proxy_smiles(frag)
        if proxy:
            parts.append(proxy)
    if not parts:
        return None
    return ".".join(parts)


def _assemble_units_to_smiles(units: List[BigSMILESUnit]) -> Optional[str]:
    if not units:
        return None
    if not RDKIT_AVAILABLE:
        return _fallback_join_smiles(units, [])

    mols = []
    for unit in units:
        mol = _mol_from_smiles_with_dummies(unit.smiles_with_dummies)
        if mol is None and unit.proxy_smiles:
            mol = _mol_from_smiles_with_dummies(unit.proxy_smiles)
        if mol is None:
            return _fallback_join_smiles(units, [])
        mols.append(mol)

    current = mols[0]
    for nxt in mols[1:]:
        merged = _connect_mols_via_dummies(current, nxt)
        if merged is None:
            return _fallback_join_smiles(units, [])
        current = merged

    current = _remove_all_dummies(current)
    try:
        smiles = Chem.MolToSmiles(current, canonical=True)
        return smiles if smiles else _fallback_join_smiles(units, [])
    except Exception:
        return _fallback_join_smiles(units, [])


def _sample_segment_units(
    blocks: List[BigSMILESBlock],
    rng: random.Random,
    min_repeat_units: int,
    max_repeat_units: int,
) -> List[BigSMILESUnit]:
    if not blocks:
        return []

    n_units = rng.randint(min_repeat_units, max_repeat_units)
    sampled: List[BigSMILESUnit] = []

    if len(blocks) == 1:
        block = blocks[0]
        if block.end_groups:
            left_cap = _weighted_choice(block.end_groups, rng)
            if left_cap is not None:
                sampled.append(left_cap)
        for _ in range(n_units):
            unit = _weighted_choice(block.repeat_units, rng)
            if unit is not None:
                sampled.append(unit)
        if block.end_groups:
            right_cap = _weighted_choice(block.end_groups, rng)
            if right_cap is not None:
                sampled.append(right_cap)
        return sampled

    for _ in range(n_units):
        block = _weighted_choice_block(blocks, rng)
        if block is None:
            continue
        unit = _weighted_choice(block.repeat_units, rng)
        if unit is not None:
            sampled.append(unit)
    return sampled


def sample_bigsmiles_realizations(
    text,
    n_samples: int = 4,
    min_repeat_units: int = 2,
    max_repeat_units: int = 6,
    random_state: Optional[int] = None,
    num_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[str]:
    if num_samples is not None:
        n_samples = num_samples
    if seed is not None and random_state is None:
        random_state = seed

    graph = parse_bigsmiles_stochastic_graph(text)
    if graph is None:
        return []

    n_samples = max(1, int(n_samples or 1))
    min_repeat_units = max(1, int(min_repeat_units or 1))
    max_repeat_units = max(min_repeat_units, int(max_repeat_units or min_repeat_units))
    rng = random.Random(random_state)

    segment_map: OrderedDict[int, List[BigSMILESBlock]] = OrderedDict()
    for block in graph.blocks:
        segment_map.setdefault(int(block.segment_id), []).append(block)

    sampled: List[str] = []
    for sample_idx in range(n_samples):
        local_rng = random.Random(rng.randint(0, 10**9) + sample_idx)
        segment_smiles: List[str] = []
        fallback_units: List[BigSMILESUnit] = []

        for _, segment_blocks in segment_map.items():
            units = _sample_segment_units(
                segment_blocks,
                rng=local_rng,
                min_repeat_units=min_repeat_units,
                max_repeat_units=max_repeat_units,
            )
            fallback_units.extend(units)
            segment_smi = _assemble_units_to_smiles(units)
            if segment_smi:
                segment_smiles.append(segment_smi)

        free_parts = [frag for frag in graph.free_fragments if frag]
        if not segment_smiles and fallback_units:
            fallback = _fallback_join_smiles(fallback_units, free_parts)
            if fallback:
                sampled.append(fallback)
            continue

        combined = ".".join([x for x in segment_smiles + free_parts if x]).strip(".")
        if combined:
            sampled.append(combined)

    deduped = list(OrderedDict((s, None) for s in sampled if s).keys())
    return deduped
