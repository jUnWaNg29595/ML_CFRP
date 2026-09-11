"""Vendored structure renderer for SMILES / BigSMILES.

Public API (kept compatible with the host app in ``app.py``):

- ``RenderOptions``  – rendering configuration dataclass
- ``RenderResult``   – per-structure outcome dataclass
- ``render_structure`` – render one structure string into ``output_dir``

SMILES molecules are drawn directly with RDKit. BigSMILES strings are first
validated with the ``bigsmiles`` package, then the stochastic fragment is
expanded into a plain SMILES chain segment and drawn with RDKit so that the
repeat-unit / end-group semantics stay visible.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D


# Bond descriptors such as [>] [<] [$] [=] [#] [~] [&] [&1] [>2] ...
_BOND_DESCRIPTOR_RE = re.compile(r"\[[<>=#$~&][0-9]*\]")
_STOCHASTIC_BLOCK_RE = re.compile(r"\{([^{}]*)\}")


@dataclass
class RenderOptions:
    """Configuration for :func:`render_structure`."""

    requested_type: str = "auto"  # "auto" | "smiles" | "bigsmiles"
    render_sample_chain: bool = False
    repeat_units: int = 5
    random_seed: int = 42
    image_width: int = 600
    image_height: int = 400


@dataclass
class RenderResult:
    """Outcome of rendering one structure string."""

    raw_string: str = ""
    detected_type: str = ""  # "smiles" | "bigsmiles" | ""
    parse_status: str = ""  # "valid" | "invalid" | "empty"
    normalized_structure: str = ""
    draw_status: str = ""  # "rendered" | "valid_but_not_renderable" | "skipped"
    error_message: str = ""
    warning_message: str = ""
    renderer: str = ""
    main_image_path: str | None = None
    sample_image_path: str | None = None
    extra: dict = field(default_factory=dict)


def detect_structure_type(raw: str) -> str:
    """Return ``"bigsmiles"`` when stochastic-object braces are present."""
    return "bigsmiles" if "{" in raw and "}" in raw else "smiles"


def _strip_descriptors(text: str) -> str:
    """Remove BigSMILES bonding descriptors and stochastic-object braces."""
    text = _BOND_DESCRIPTOR_RE.sub("", text)
    text = _STOCHASTIC_BLOCK_RE.sub(lambda m: m.group(1), text)
    return text.strip()


def _expand_repeat_units(text: str, repeat_units: int) -> str:
    """Repeat every stochastic fragment ``repeat_units`` times to fake a chain."""
    if repeat_units <= 1:
        return text
    return _STOCHASTIC_BLOCK_RE.sub(lambda m: m.group(1) * int(repeat_units), text)


def _mol_from_smiles(text: str):
    return Chem.MolFromSmiles(text)


def _draw_mol_to_png(mol, path: Path, width: int, height: int, legend: str = "") -> None:
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(int(width), int(height))
    if legend:
        drawer.drawOptions().legendFontSize = max(14, int(min(width, height) * 0.045))
    drawer.DrawMolecule(mol, legend=legend)
    drawer.FinishDrawing()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(drawer.GetDrawingText())


def _stable_tag(raw: str) -> str:
    return hashlib.md5(raw.encode("utf-8", "ignore")).hexdigest()[:10]


def render_structure(raw: str, output_dir, options: RenderOptions | None = None) -> RenderResult:
    """Parse and render one SMILES / BigSMILES string into ``output_dir``.

    Images are written under ``output_dir / images`` and referenced in
    :attr:`RenderResult.main_image_path` / :attr:`RenderResult.sample_image_path`
    as paths relative to ``output_dir``.
    """
    options = options or RenderOptions()
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"

    result = RenderResult(raw_string=str(raw or "").strip())

    if not result.raw_string:
        result.parse_status = "empty"
        result.draw_status = "skipped"
        return result

    requested = (options.requested_type or "auto").lower()
    detected = detect_structure_type(result.raw_string)
    if requested in ("smiles", "bigsmiles"):
        detected = requested
    result.detected_type = detected

    # ---------- parse & validate ----------
    render_smiles = ""
    if detected == "bigsmiles":
        try:
            from bigsmiles import BigSMILES  # optional dependency, imported lazily

            BigSMILES(result.raw_string)  # validation only; canonical output unused
        except ImportError:
            result.parse_status = "invalid"
            result.draw_status = "skipped"
            result.error_message = "未安装 bigsmiles 包，无法解析 BigSMILES（pip install bigsmiles）"
            return result
        except Exception as exc:  # bigsmiles.errors.BigSMILESError and friends
            result.parse_status = "invalid"
            result.draw_status = "skipped"
            result.error_message = f"BigSMILES 语法无效：{exc}"
            return result

        candidate = _strip_descriptors(result.raw_string)
        if options.render_sample_chain:
            candidate = _expand_repeat_units(candidate, max(int(options.repeat_units), 1))
        if not candidate:
            result.parse_status = "valid"
            result.draw_status = "valid_but_not_renderable"
            result.normalized_structure = result.raw_string
            result.renderer = "bigsmiles"
            result.warning_message = "随机片段展开后为空，无法用 RDKit 绘图"
            return result
        if _mol_from_smiles(candidate) is None:
            result.parse_status = "valid"
            result.draw_status = "valid_but_not_renderable"
            result.normalized_structure = result.raw_string
            result.renderer = "bigsmiles"
            result.warning_message = (
                "BigSMILES 解析成功，但展开的链段无法转换为可绘制的普通分子；"
                f"展开式：{candidate}"
            )
            return result
        render_smiles = candidate
        result.renderer = "bigsmiles+rdkit"
        result.normalized_structure = result.raw_string
    else:
        mol = _mol_from_smiles(result.raw_string)
        if mol is None:
            result.parse_status = "invalid"
            result.draw_status = "skipped"
            result.error_message = "SMILES 语法无效（RDKit 无法解析）"
            return result
        render_smiles = Chem.MolToSmiles(mol)
        result.normalized_structure = render_smiles
        result.renderer = "rdkit"

    result.parse_status = "valid"

    # ---------- draw ----------
    tag = _stable_tag(result.raw_string + "|" + render_smiles)
    try:
        mol = _mol_from_smiles(render_smiles)
        if mol is None:  # defensive; should not happen after validation above
            result.draw_status = "valid_but_not_renderable"
            return result
        main_rel = f"images/main_{tag}.png"
        _draw_mol_to_png(mol, output_dir / main_rel, options.image_width, options.image_height)
        result.main_image_path = main_rel
        result.draw_status = "rendered"
    except Exception as exc:
        result.draw_status = "valid_but_not_renderable"
        result.error_message = f"绘图失败：{exc}"
        return result

    # ---------- optional sample-chain image ----------
    if detected == "bigsmiles" and options.render_sample_chain:
        try:
            sample_smiles = _strip_descriptors(
                _expand_repeat_units(result.raw_string, max(int(options.repeat_units), 1))
            )
            sample_mol = _mol_from_smiles(sample_smiles)
            if sample_mol is not None:
                sample_rel = f"images/sample_{tag}.png"
                _draw_mol_to_png(
                    sample_mol,
                    output_dir / sample_rel,
                    max(int(options.image_width), 800),
                    max(int(options.image_height), 400),
                    legend=f"采样链段 ×{int(options.repeat_units)}",
                )
                result.sample_image_path = sample_rel
        except Exception as exc:
            result.warning_message = (
                (result.warning_message + "；" if result.warning_message else "")
                + f"采样链段绘制失败：{exc}"
            )

    return result
