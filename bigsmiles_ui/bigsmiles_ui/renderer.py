from __future__ import annotations

import hashlib
import importlib
import io
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:
    from PIL import Image, ImageChops, ImageDraw, ImageFont
except Exception:  # pragma: no cover - 由结果对象给出缺少绘图依赖的提示
    Image = None
    ImageChops = None
    ImageDraw = None
    ImageFont = None

try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import Draw, rdDepictor
except Exception:  # pragma: no cover - 运行环境缺少 RDKit 时由结果对象给出提示
    rdkit = None
    Chem = None
    Draw = None
    rdDepictor = None

ParseStatus = Literal["valid", "invalid", "empty"]
DrawStatus = Literal[
    "rendered",
    "valid_but_not_renderable",
    "render_failed",
    "main_rendered_sample_failed",
    "not_requested",
]

_ADDED_COLUMNS = (
    "结构检查_原始字符串",
    "结构检查_识别类型",
    "结构检查_解析状态",
    "结构检查_规范化结构",
    "结构检查_主图路径",
    "结构检查_采样图路径",
    "结构检查_绘图状态",
    "结构检查_错误信息",
    "结构检查_警告信息",
    "结构检查_渲染器",
)
_BIGSMILES_MARKER_RE = re.compile(r"\{|\}|\[\s*[$<>]")
_BIGSMILES_BOND_SYMBOLS = frozenset("-=#:~/\\")
_BIGSMILES_ORGANIC_ATOMS = frozenset("BCNOPSFIbcnops")
_BIGSMILES_TWO_LETTER_ATOMS = frozenset(
    {"Cl", "Br", "Si", "Se", "Na", "Li", "Al", "Ca", "Mg", "Fe", "Zn", "Cu", "Ag", "Sn", "As"}
)


@dataclass(frozen=True)
class RenderOptions:
    requested_type: str = "auto"
    render_sample_chain: bool = False
    repeat_units: int = 5
    random_seed: int = 42
    image_width: int = 1000
    image_height: int = 700
    # ----------------------------------------------------------------
    # 画质与展示选项（新增，默认值与旧行为保持一致）
    # ----------------------------------------------------------------
    #: 超采样倍数。1.0 表示与 image_width/height 完全一致；2.0 表示按两倍像素
    #: 渲染，笔画更细、边缘更锐利，浏览器缩小时依然清晰。
    supersample: float = 1.0
    #: 是否根据分子二维布局的长宽比自动收紧画布（把多余留白裁掉）。
    auto_fit: bool = False
    #: 是否把顶层 "." 连接的多组分拆成独立面板（并附一张整体视图）。
    split_components: bool = True
    #: 是否把 BigSMILES 的 {重复单元} 就地展开成一张连通的完整骨架图。
    expand_repeat_units: bool = True
    #: 纯 SMILES 路径是否额外输出矢量 SVG。
    emit_svg: bool = True

    def __post_init__(self) -> None:
        requested = str(self.requested_type).strip().lower()
        if requested not in {"auto", "smiles", "bigsmiles"}:
            raise ValueError("结构类型必须是 auto、smiles 或 bigsmiles")
        if not 1 <= int(self.repeat_units) <= 50:
            raise ValueError("重复单元数必须在 1 到 50 之间")
        if not 200 <= int(self.image_width) <= 3000:
            raise ValueError("图片宽度必须在 200 到 3000 之间")
        if not 200 <= int(self.image_height) <= 3000:
            raise ValueError("图片高度必须在 200 到 3000 之间")
        if not 1.0 <= float(self.supersample) <= 4.0:
            raise ValueError("超采样倍数必须在 1.0 到 4.0 之间")


@dataclass
class RenderResult:
    raw_string: str
    detected_type: str
    parse_status: ParseStatus
    normalized_structure: str = ""
    main_image_path: str = ""
    sample_image_path: str = ""
    draw_status: DrawStatus = "not_requested"
    error_message: str = ""
    warning_message: str = ""
    renderer: str = ""
    #: 纯 SMILES 路径下与主图同名的矢量 SVG 相对路径（无则空串）。
    #: 故意不进入 to_scalar_dict，避免破坏批量结果表的固定列契约。
    svg_path: str = ""
    #: 与具体输入无关的固定免责说明；与真实问题分开，避免互相淹没。
    #: 同样不进入 to_scalar_dict。
    disclaimers: tuple[str, ...] = ()
    #: 官方 BigSMILES 解析器状态：accepted / rejected / unavailable / ""（未走该路径）。
    parser_status: str = ""

    def to_scalar_dict(self) -> dict[str, object]:
        values: dict[str, object] = {
            "结构检查_原始字符串": self.raw_string,
            "结构检查_识别类型": self.detected_type,
            "结构检查_解析状态": self.parse_status,
            "结构检查_规范化结构": self.normalized_structure,
            "结构检查_主图路径": self.main_image_path,
            "结构检查_采样图路径": self.sample_image_path,
            "结构检查_绘图状态": self.draw_status,
            "结构检查_错误信息": self.error_message,
            "结构检查_警告信息": self.warning_message,
            "结构检查_渲染器": self.renderer,
        }
        for key, value in values.items():
            if isinstance(value, (list, dict, tuple, set)):
                raise TypeError(f"{key} 必须是标量值")
        return values


def added_columns() -> tuple[str, ...]:
    return _ADDED_COLUMNS


def identify_structure_type(raw: str, requested_type: str = "auto") -> str:
    text = "" if raw is None else str(raw)
    requested = str(requested_type or "auto").strip().lower()
    if requested in {"smiles", "bigsmiles"}:
        return requested if text.strip() else "unknown"
    if requested != "auto":
        raise ValueError("结构类型必须是 auto、smiles 或 bigsmiles")
    if not text.strip():
        return "unknown"
    if _BIGSMILES_MARKER_RE.search(text):
        return "bigsmiles"
    return "smiles"


def _rdkit_renderer_name() -> str:
    if rdkit is None:
        return "RDKit（未安装）"
    return f"RDKit {getattr(rdkit, '__version__', '未知版本')}"


def _hash_name(text: str, suffix: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{digest}_{suffix}.png"


def _ensure_output_dir(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _invalid_result(raw: str, detected_type: str, message: str, renderer: str) -> RenderResult:
    return RenderResult(
        raw_string=raw,
        detected_type=detected_type,
        parse_status="invalid",
        draw_status="not_requested",
        error_message=message,
        renderer=renderer,
    )


# ---------------------------------------------------------------------------
# 画布工具：超采样、自适应画布、多面板 PNG + SVG 双输出
# ---------------------------------------------------------------------------

#: SMILES 文本原子 token 统计（用于把展开后的字符区间映射为原子下标）。
_SMILES_ATOM_TOKEN_RE = re.compile(r"\[[^\]]*\]|Br|Cl|B|C|N|O|P|S|F|I|b|c|n|o|p|s")
#: RDKit 高亮色（0~1 浮点 RGB）。
_HIGHLIGHT_COLOUR = (1.0, 0.86, 0.36)
#: SVG 里的中文字体栈（RDKit 无法渲染 CJK，所以文字一律由我们自己输出）。
_SVG_FONT_STACK = (
    "'Microsoft YaHei','PingFang SC','Noto Sans CJK SC','Source Han Sans SC',"
    "'WenQuanYi Zen Hei','Heiti SC',sans-serif"
)


def _draw_api():
    """返回 ``rdMolDraw2D``；RDKit 不可用时返回 None。"""
    if Draw is None or Chem is None:
        return None
    try:
        from rdkit.Chem.Draw import rdMolDraw2D
    except Exception:
        return None
    return rdMolDraw2D


def _supersample_scale(options: RenderOptions) -> float:
    try:
        value = float(getattr(options, "supersample", 1.0) or 1.0)
    except Exception:
        return 1.0
    return max(1.0, min(4.0, value))


def _scaled(value: float, scale: float) -> int:
    return int(round(float(value) * scale))


def _prepare_drawable(molecule):
    """复制分子并补齐二维坐标，避免污染调用方对象。"""
    if Chem is None or molecule is None:
        return molecule
    drawable = Chem.Mol(molecule)
    if rdDepictor is not None:
        try:
            rdDepictor.Compute2DCoords(drawable, canonOrient=True)
        except Exception:
            pass
    return drawable


def _molecule_aspect_ratio(molecule) -> float | None:
    """二维坐标包围盒的宽高比；无法判断时返回 None。"""
    drawable = _prepare_drawable(molecule)
    if drawable is None:
        return None
    try:
        conformer = drawable.GetConformer()
    except Exception:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for index in range(drawable.GetNumAtoms()):
        point = conformer.GetAtomPosition(index)
        xs.append(point.x)
        ys.append(point.y)
    if not xs:
        return None
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if width <= 1e-6 or height <= 1e-6:
        return None
    return width / height


def _fit_canvas_size(molecules: list, options: RenderOptions) -> tuple[int, int]:
    """按分子二维布局的长宽比收紧画布，避免小分子被大片留白淹没。

    仅在 ``options.auto_fit`` 打开时生效；此时 ``image_height`` 被当作上限。
    """
    width = int(options.image_width)
    height = int(options.image_height)
    if not getattr(options, "auto_fit", False) or not molecules:
        return width, height
    aspects = [value for value in (_molecule_aspect_ratio(mol) for mol in molecules) if value]
    if not aspects:
        return width, height
    # 原子坐标不含原子标签与氢原子文字，留 18% 余量后再交给 RDKit 的 padding。
    fitted = int(round(width / (min(aspects) * 1.18)))
    return width, max(200, min(height, fitted))


def _configure_draw_options(drawer, scale: float) -> None:
    """统一控制留白、线宽与字号，保证放大后依然锐利、紧凑。"""
    try:
        draw_options = drawer.drawOptions()
    except Exception:
        return
    settings = (
        ("padding", 0.02),
        ("bondLineWidth", max(1.6, 2.0 * scale)),
        ("minFontSize", max(11.0, 13.0 * scale)),
        ("maxFontSize", max(15.0, 20.0 * scale)),
        ("legendFontSize", max(14.0, 17.0 * scale)),
        ("highlightRadius", 0.42),
        ("highlightColour", _HIGHLIGHT_COLOUR),
    )
    for name, value in settings:
        try:
            setattr(draw_options, name, value)
        except Exception:
            continue


def _molecule_png(molecule, pixel_size: tuple[float, float], scale: float, highlights=()) -> "Image.Image | None":
    """在给定像素尺寸内用 RDKit Cairo 绘制单个分子，返回 PIL 图像。

    ``pixel_size`` 已经是设备像素（调用方已乘过 ``scale``），这里只把 ``scale``
    用于线宽/字号等绘制参数。
    """
    draw_module = _draw_api()
    if draw_module is None or Image is None or molecule is None:
        return None
    width = max(120, int(round(float(pixel_size[0]))))
    height = max(120, int(round(float(pixel_size[1]))))
    try:
        drawer = draw_module.MolDraw2DCairo(width, height)
        _configure_draw_options(drawer, scale)
        drawer.DrawMolecule(
            _prepare_drawable(molecule),
            highlightAtoms=list(highlights) or None,
        )
        drawer.FinishDrawing()
        payload = drawer.GetDrawingText()
    except Exception:
        return None
    try:
        return Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception:
        return None


def _molecule_svg_fragment(molecule, pixel_size: tuple[float, float], scale: float, highlights=()) -> str:
    """返回可嵌入外层 SVG 的分子图形片段（不含 xml 声明与根 svg 标签）。"""
    draw_module = _draw_api()
    if draw_module is None or molecule is None:
        return ""
    width = max(120, int(round(float(pixel_size[0]))))
    height = max(120, int(round(float(pixel_size[1]))))
    try:
        drawer = draw_module.MolDraw2DSVG(width, height)
        _configure_draw_options(drawer, scale)
        drawer.DrawMolecule(
            _prepare_drawable(molecule),
            highlightAtoms=list(highlights) or None,
        )
        drawer.FinishDrawing()
        document = drawer.GetDrawingText()
    except Exception:
        return ""
    match = re.search(r"<svg[^>]*>", document)
    if not match:
        return ""
    body = document[match.end():]
    end = body.rfind("</svg>")
    return body[:end] if end >= 0 else ""


def _svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _svg_text(x: float, y: float, text: str, *, size: int, color: str, bold: bool = False) -> str:
    weight = "700" if bold else "400"
    return (
        f"<text x='{x:.1f}' y='{y:.1f}' font-size='{int(size)}px' font-weight='{weight}' "
        f"fill='{color}' font-family=\"{_SVG_FONT_STACK}\">{_svg_escape(text)}</text>"
    )


def _rgb_to_hex(colour) -> str:
    try:
        red, green, blue = (int(channel) for channel in colour)
    except Exception:
        return "#f8fafc"
    return f"#{max(0, min(255, red)):02x}{max(0, min(255, green)):02x}{max(0, min(255, blue)):02x}"


@dataclass
class _Panel:
    """一个面板：标题、脚注、可选分子与高亮原子。"""

    title: str = ""
    note: str = ""
    molecule: object | None = None
    highlight_atoms: tuple[int, ...] = ()
    fill: tuple[int, int, int] = (248, 250, 252)
    outline: tuple[int, int, int] = (150, 160, 170)
    title_color: tuple[int, int, int] = (70, 80, 90)
    empty_text: str = "该片段无法由 RDKit 解析"
    empty_color: tuple[int, int, int] = (170, 60, 50)


def _panel_heights(rows: list[list[_Panel]], usable_width: float, gap: float, inner_pad: float,
                   title_h: float, note_h: float, min_cell_h: float, max_cell_h: float) -> tuple[list[float], list[float]]:
    """按每行内容的长宽比计算行高，避免宽分子被正方形面板压小。"""
    heights: list[float] = []
    widths: list[float] = []
    for row in rows:
        count = max(1, len(row))
        cell_w = (usable_width - gap * (count - 1)) / count
        widths.append(cell_w)
        inner_w = max(80.0, cell_w - 2 * inner_pad)
        aspects = [
            value
            for value in (_molecule_aspect_ratio(panel.molecule) for panel in row)
            if value
        ]
        if not aspects:
            heights.append(max(min_cell_h, title_h + note_h + inner_w * 0.5 + 2 * inner_pad))
            continue
        # 几何均值：宽分子与紧凑分子同处一行时取折中高度，既不会把宽分子压小，
        # 也不会为紧凑分子留下大片空白。
        mean_aspect = math.exp(sum(math.log(max(0.5, min(12.0, a))) for a in aspects) / len(aspects))
        content_h = inner_w / mean_aspect
        heights.append(
            max(min_cell_h, min(max_cell_h, title_h + note_h + 2 * inner_pad + content_h))
        )
    return heights, widths


def _compose_panel_figure(
    rows: list[list[_Panel]],
    output_png: Path,
    *,
    title: str = "",
    subtitle: str = "",
    footer: str = "",
    width: int = 1000,
    scale: float = 1.0,
    output_svg: Path | None = None,
    min_cell_h: float = 200.0,
    max_cell_h: float = 460.0,
) -> tuple[int, int]:
    """把若干行面板合成一张 PNG（并可选输出同版式的 SVG）。

    高度按内容自适应（不再强行填满调用方给的画布），因此不会出现“分子很小”
    或面板被裁掉的情况。返回实际生成的像素尺寸。
    """
    if Image is None or ImageDraw is None:
        raise RuntimeError("当前 Python 环境未安装 Pillow，无法合成结构图")
    rows = [row for row in rows if row]
    if not rows:
        raise ValueError("没有可绘制的面板")

    margin_x = _scaled(32, scale)
    gap = _scaled(18, scale)
    inner_pad = _scaled(14, scale)
    title_h = _scaled(38, scale) if any(panel.title for row in rows for panel in row) else 0
    note_h = _scaled(30, scale) if any(panel.note for row in rows for panel in row) else 0
    header_h = _scaled(92, scale) if (title or subtitle) else _scaled(18, scale)
    footer_h = _scaled(48, scale) if footer else _scaled(18, scale)

    total_w = _scaled(width, scale)
    usable_w = max(200.0, total_w - 2 * margin_x)
    row_heights, _ = _panel_heights(
        rows, usable_w, gap, inner_pad, title_h, note_h,
        _scaled(min_cell_h, scale), _scaled(max_cell_h, scale),
    )
    total_h = int(round(header_h + sum(row_heights) + gap * (len(rows) - 1) + footer_h))

    image = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(image)
    title_font = _load_schematic_font(int(25 * scale), bold=True)
    subtitle_font = _load_schematic_font(int(13 * scale))
    label_font = _load_schematic_font(int(16 * scale), bold=True)
    note_font = _load_schematic_font(int(13 * scale))
    empty_font = _load_schematic_font(int(13 * scale))

    if title:
        draw.text((margin_x, _scaled(18, scale)), title, fill=(28, 45, 60), font=title_font)
    if subtitle:
        draw.text((margin_x, _scaled(54, scale)), subtitle, fill=(80, 90, 100), font=subtitle_font)

    svg_parts: list[str] = []
    if output_svg is not None:
        svg_parts.append("<?xml version='1.0' encoding='UTF-8'?>")
        svg_parts.append(
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{total_w}px' height='{total_h}px' "
            f"viewBox='0 0 {total_w} {total_h}'>"
        )
        svg_parts.append(f"<rect width='{total_w}' height='{total_h}' fill='#ffffff'/>")
        if title:
            svg_parts.append(_svg_text(margin_x, _scaled(18, scale) + 25 * scale, title,
                                       size=25 * scale, color="#1c2d3c", bold=True))
        if subtitle:
            svg_parts.append(_svg_text(margin_x, _scaled(54, scale) + 14 * scale, subtitle,
                                       size=13 * scale, color="#505a64"))

    cursor_y = float(header_h)
    for row, cell_h in zip(rows, row_heights):
        count = max(1, len(row))
        cell_w = (usable_w - gap * (count - 1)) / count
        for column, panel in enumerate(row):
            left = margin_x + column * (cell_w + gap)
            top = cursor_y
            right, bottom = left + cell_w, top + cell_h
            draw.rounded_rectangle(
                (left, top, right, bottom), radius=max(6, int(10 * scale)),
                fill=panel.fill, outline=panel.outline, width=max(2, int(2 * scale)),
            )
            if output_svg is not None:
                svg_parts.append(
                    f"<rect x='{left:.1f}' y='{top:.1f}' width='{cell_w:.1f}' height='{cell_h:.1f}' "
                    f"rx='{max(6, int(10 * scale))}' fill='{_rgb_to_hex(panel.fill)}' "
                    f"stroke='{_rgb_to_hex(panel.outline)}' stroke-width='{max(2, int(2 * scale))}'/>"
                )
            if panel.title:
                draw.text((left + inner_pad, top + _scaled(8, scale)), panel.title,
                          fill=panel.title_color, font=label_font)
                if output_svg is not None:
                    svg_parts.append(_svg_text(
                        left + inner_pad, top + _scaled(8, scale) + 16 * scale, panel.title,
                        size=16 * scale, color=_rgb_to_hex(panel.title_color), bold=True,
                    ))

            inner_w = max(120.0, cell_w - 2 * inner_pad)
            inner_h = max(100.0, cell_h - title_h - note_h - inner_pad)
            inner_x = left + inner_pad
            inner_y = top + title_h
            if panel.molecule is not None:
                molecule_image = _molecule_png(
                    panel.molecule, (inner_w, inner_h), scale, panel.highlight_atoms
                )
                if molecule_image is not None:
                    image.paste(molecule_image, (int(inner_x), int(inner_y)))
                if output_svg is not None:
                    fragment = _molecule_svg_fragment(
                        panel.molecule, (inner_w, inner_h), scale, panel.highlight_atoms
                    )
                    if fragment:
                        svg_parts.append(f"<g transform='translate({inner_x:.1f},{inner_y:.1f})'>")
                        svg_parts.append(fragment)
                        svg_parts.append("</g>")
            else:
                draw.text((left + inner_pad, top + cell_h / 2), panel.empty_text,
                          fill=panel.empty_color, font=empty_font)
                if output_svg is not None:
                    svg_parts.append(_svg_text(
                        left + inner_pad, top + cell_h / 2, panel.empty_text,
                        size=13 * scale, color=_rgb_to_hex(panel.empty_color),
                    ))
            if panel.note:
                draw.text((left + inner_pad, bottom - note_h), panel.note,
                          fill=(96, 105, 115), font=note_font)
                if output_svg is not None:
                    svg_parts.append(_svg_text(
                        left + inner_pad, bottom - note_h + 13 * scale, panel.note,
                        size=13 * scale, color="#606973",
                    ))
        cursor_y += cell_h + gap

    if footer:
        draw.text((margin_x, total_h - _scaled(34, scale)), footer, fill=(90, 90, 90), font=note_font)
        if output_svg is not None:
            svg_parts.append(_svg_text(
                margin_x, total_h - _scaled(34, scale) + 13 * scale, footer,
                size=13 * scale, color="#5a5a5a",
            ))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png, format="PNG")
    if output_svg is not None:
        svg_parts.append("</svg>")
        try:
            output_svg.parent.mkdir(parents=True, exist_ok=True)
            output_svg.write_text("\n".join(svg_parts), encoding="utf-8")
        except Exception:
            pass
    return total_w, total_h


def _molecule_caption(molecule) -> str:
    """分子式 + 相对分子质量（用于多组分面板脚注）。"""
    if molecule is None:
        return ""
    try:
        from rdkit.Chem import rdMolDescriptors

        formula = rdMolDescriptors.CalcMolFormula(molecule)
        weight = rdMolDescriptors.CalcExactMolWt(molecule)
        return f"{formula} · MW {weight:.2f}"
    except Exception:
        return ""


def _render_smiles_grid(
    components: list,
    whole_molecule,
    output_dir: Path,
    options: RenderOptions,
    image_name: str,
) -> tuple[bool, str]:
    """多组分（顶层 "." 连接）结构图：逐组分面板 + 整体视图。

    返回 ``(是否成功, SVG 相对路径)``。所有中文都由 Pillow/自建 SVG 输出，
    因为 RDKit 自身无法绘制 CJK 字形。
    """
    scale = _supersample_scale(options)
    panels: list[_Panel] = []
    for index, molecule in enumerate(components):
        panels.append(
            _Panel(
                title=f"组分 {index + 1} / {len(components)}",
                note=_molecule_caption(molecule),
                molecule=molecule,
                fill=(246, 249, 253),
                outline=(83, 116, 150),
                title_color=(45, 82, 125),
            )
        )
    rows: list[list[_Panel]] = []
    per_row = 3 if len(panels) >= 3 else max(1, len(panels))
    for start in range(0, len(panels), per_row):
        rows.append(panels[start:start + per_row])
    rows.append(
        [
            _Panel(
                title=f"整体视图（{len(components)} 个组分，以 . 连接）",
                note=_molecule_caption(whole_molecule),
                molecule=whole_molecule,
                fill=(255, 250, 229),
                outline=(0, 121, 107),
                title_color=(0, 105, 92),
            )
        ]
    )
    svg_name = Path(image_name).with_suffix(".svg").name if options.emit_svg else None
    try:
        _compose_panel_figure(
            rows,
            output_dir / image_name,
            title=f"多组分结构图（{len(components)} 个组分）",
            subtitle="按顶层“.”拆分：每个组分单独成图，最后一行是全部组分的整体布局。",
            width=int(options.image_width),
            scale=scale,
            output_svg=(output_dir / svg_name) if svg_name else None,
        )
    except Exception:
        return False, ""
    return True, (svg_name or "")


def _content_bbox(image, background=(255, 255, 255)):
    """返回非背景内容的包围盒；全白时返回 None。"""
    if Image is None or ImageChops is None:
        return None
    try:
        background_image = Image.new("RGB", image.size, background)
        return ImageChops.difference(image.convert("RGB"), background_image).getbbox()
    except Exception:
        return None


def _trim_content(image, margin: int = 0):
    """裁掉四周留白（保留 ``margin`` 像素白边）。"""
    box = _content_bbox(image)
    if box is None:
        return image
    left, top, right, bottom = box
    return image.crop(
        (
            max(0, left - margin),
            max(0, top - margin),
            min(image.width, right + margin),
            min(image.height, bottom + margin),
        )
    )


def _render_single_smiles(
    molecule,
    options: RenderOptions,
) -> tuple["Image.Image | None", tuple[int, int]]:
    """单分子高清渲染。

    开启 ``auto_fit`` 时采用两遍法：先按估算长宽比试画一次，量出真实内容包围盒，
    再按真实长宽比重画。这样原子标签、氢原子文字都会计入，分子能真正填满画布，
    而不是被正方形画布掏空。
    """
    scale = _supersample_scale(options)
    width = int(options.image_width)
    max_height = int(options.image_height)
    if not getattr(options, "auto_fit", False):
        image = _molecule_png(molecule, (width * scale, max_height * scale), scale)
        return image, (width, max_height)

    probe_height = _fit_canvas_size([molecule], options)[1]
    probe = _molecule_png(molecule, (width * scale, probe_height * scale), scale)
    if probe is None:
        return None, (width, max_height)

    box = _content_bbox(probe)
    if box is None:
        return probe, (width, probe_height)
    left, top, right, bottom = box
    content_area = max(0, right - left) * max(0, bottom - top)
    if content_area < 0.03 * float(probe.width * probe.height):
        # 内容极小（如单原子、“[Na+].[Cl-]”），强行拉满只会得到一张巨大的字号；
        # 此时回退到用户给定的画布尺寸。
        image = _molecule_png(molecule, (width * scale, max_height * scale), scale)
        return image or probe, (width, max_height)

    true_aspect = (right - left) / max(1, bottom - top)
    fitted_height = int(round(width / max(0.05, true_aspect)))
    fitted_height = max(200, min(max_height, fitted_height))
    if abs(fitted_height - probe_height) <= 2:
        return probe, (width, probe_height)
    image = _molecule_png(molecule, (width * scale, fitted_height * scale), scale)
    if image is None:
        return probe, (width, probe_height)
    return image, (width, fitted_height)


def _render_molecules(
    molecules: list,
    output_path: Path,
    options: RenderOptions,
    legends: list[str] | None = None,
) -> None:
    """单分子/多分子图像输出（保留旧签名，内部改走高清 Cairo 通道）。"""
    if not molecules:
        raise ValueError("没有可绘制的分子对象")
    if Draw is None:
        raise RuntimeError("当前 Python 环境未安装 RDKit，无法绘图")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = int(options.image_width), int(options.image_height)
    if len(molecules) == 1:
        image, _size = _render_single_smiles(molecules[0], options)
        if image is not None:
            image.save(output_path, format="PNG")
            return
    image = Draw.MolsToGridImage(
        molecules,
        molsPerRow=min(len(molecules), 4),
        subImgSize=(max(200, width // min(len(molecules), 4)), max(200, height // 2)),
        legends=legends,
        useSVG=False,
    )
    image.save(output_path, format="PNG")


def _smiles_components(cleaned: str, molecule) -> list:
    """按顶层 "." 拆分并逐个解析；任何一段解析失败则退化为整体。"""
    if Chem is None or molecule is None:
        return [molecule] if molecule is not None else []
    pieces = [part for part in _split_top_level(cleaned, ".") if part.strip()]
    if len(pieces) < 2:
        return [molecule]
    parsed: list = []
    for piece in pieces:
        try:
            component = Chem.MolFromSmiles(piece)
        except Exception:
            component = None
        if component is None:
            return [molecule]
        parsed.append(component)
    return parsed


def _render_smiles(raw: str, output_dir: Path, options: RenderOptions) -> RenderResult:
    renderer = _rdkit_renderer_name()
    if Chem is None:
        return _invalid_result(raw, "smiles", "当前 Python 环境未安装 RDKit", renderer)
    cleaned = raw.strip()
    try:
        molecule = Chem.MolFromSmiles(cleaned)
    except Exception as exc:
        return _invalid_result(raw, "smiles", f"SMILES 解析失败：{exc}", renderer)
    if molecule is None:
        return _invalid_result(raw, "smiles", "SMILES 解析失败：结构语法无效", renderer)
    try:
        normalized = Chem.MolToSmiles(molecule, canonical=True)
    except Exception as exc:
        return _invalid_result(raw, "smiles", f"SMILES 规范化失败：{exc}", renderer)

    image_name = _hash_name(cleaned, "main")
    warnings: list[str] = []
    components = _smiles_components(cleaned, molecule) if options.split_components else [molecule]
    use_grid = len(components) > 1 and options.split_components
    svg_name = ""
    try:
        if use_grid:
            ok, svg_name = _render_smiles_grid(components, molecule, output_dir, options, image_name)
            if not ok:
                use_grid = False
                components = [molecule]
        if not use_grid:
            scale = _supersample_scale(options)
            image, (size_w, size_h) = _render_single_smiles(molecule, options)
            if image is None:
                raise RuntimeError("RDKit 绘图通道不可用")
            output_dir.mkdir(parents=True, exist_ok=True)
            image.save(output_dir / image_name, format="PNG")
            if options.emit_svg:
                fragment = _molecule_svg_fragment(
                    molecule, (size_w * scale, size_h * scale), scale
                )
                if fragment:
                    candidate = Path(image_name).with_suffix(".svg").name
                    svg_document = (
                        "<?xml version='1.0' encoding='UTF-8'?>\n"
                        f"<svg xmlns='http://www.w3.org/2000/svg' width='{size_w * scale:.0f}px' "
                        f"height='{size_h * scale:.0f}px' viewBox='0 0 {size_w * scale:.0f} {size_h * scale:.0f}'>\n"
                        f"<rect width='100%' height='100%' fill='#ffffff'/>\n{fragment}\n</svg>\n"
                    )
                    try:
                        (output_dir / candidate).write_text(svg_document, encoding="utf-8")
                        svg_name = candidate
                    except Exception:
                        svg_name = ""
    except Exception as exc:
        return RenderResult(
            raw_string=raw,
            detected_type="smiles",
            parse_status="valid",
            normalized_structure=normalized,
            draw_status="render_failed",
            error_message=f"SMILES 已解析，但绘图失败：{exc}",
            renderer=renderer,
        )

    if use_grid:
        warnings.append(
            f"检测到 {len(components)} 个顶层组分（以 . 连接）：已逐组分出图并附整体视图"
        )
    return RenderResult(
        raw_string=raw,
        detected_type="smiles",
        parse_status="valid",
        normalized_structure=normalized,
        main_image_path=image_name,
        draw_status="rendered",
        warning_message="；".join(warnings),
        renderer=renderer,
        svg_path=svg_name,
    )


@dataclass
class _BigSMILESParse:
    valid: bool
    normalized: str
    fragments: list[str]
    renderer: str
    warning: str = ""
    error: str = ""
    parser_status: str = ""


def _balanced_bigsmiles(text: str) -> bool:
    pairs = {"[": "]", "(": ")", "{": "}"}
    stack: list[str] = []
    for char in text:
        if char in pairs:
            stack.append(char)
        elif char in pairs.values():
            if not stack or pairs[stack.pop()] != char:
                return False
    return not stack


def _split_top_level(text: str, separator: str = ".") -> list[str]:
    result: list[str] = []
    start = 0
    stack: list[str] = []
    pairs = {"[": "]", "(": ")", "{": "}"}
    for index, char in enumerate(text):
        if char in pairs:
            stack.append(char)
        elif char in pairs.values() and stack:
            stack.pop()
        elif char == separator and not stack:
            result.append(text[start:index])
            start = index + 1
    result.append(text[start:])
    return result


def _extract_braced_objects(text: str) -> list[str]:
    fragments: list[str] = []
    start: int | None = None
    depth = 0
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index + 1
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                fragments.extend(part for part in _split_top_level(text[start:index]) if part.strip())
                start = None
    return fragments


def _remove_bigsmiles_bracket_token(text: str) -> str:
    output: list[str] = []
    index = 0
    while index < len(text):
        if text[index] != "[":
            output.append(text[index])
            index += 1
            continue
        depth = 1
        cursor = index + 1
        while cursor < len(text) and depth:
            if text[cursor] == "[":
                depth += 1
            elif text[cursor] == "]":
                depth -= 1
            cursor += 1
        if depth:
            output.append(text[index:])
            break
        token = text[index + 1 : cursor - 1].strip()
        if not token.startswith(("$", "<", ">")) and token != "*":
            output.append(text[index:cursor])
        index = cursor
    return "".join(output)


def _candidate_fragments(text: str) -> list[str]:
    candidates = _extract_braced_objects(text)
    if not candidates:
        candidates = [text]
    cleaned: list[str] = []
    for candidate in candidates:
        candidate = _remove_bigsmiles_bracket_token(candidate)
        candidate = candidate.replace("{", "").replace("}", "")
        candidate = re.sub(r"\s+", "", candidate)
        candidate = candidate.strip(".-")
        if candidate and candidate not in cleaned:
            cleaned.append(candidate)
    return cleaned


@dataclass
class _SchematicNode:
    text: str
    kind: str
    x: float
    y: float
    ring_labels: list[str]


@dataclass
class _SchematicEdge:
    start: int
    end: int
    bond: str = "-"


@dataclass
class _SchematicGraph:
    nodes: list[_SchematicNode]
    edges: list[_SchematicEdge]
    annotations: list[tuple[str, float, float]]
    warnings: list[str]


def _find_matching_token(text: str, start: int, opener: str, closer: str) -> int | None:
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return index
    return None


def _split_bigsmiles_sections(text: str) -> list[tuple[str, str]]:
    """Split top-level chain text and stochastic objects without interpreting chemistry."""
    sections: list[tuple[str, str]] = []
    segment_start = 0
    index = 0
    while index < len(text):
        if text[index] != "{":
            index += 1
            continue
        close = _find_matching_token(text, index, "{", "}")
        if close is None:
            break
        segment = text[segment_start:index].strip()
        if segment:
            sections.append(("segment", segment))
        inner = text[index + 1 : close].strip()
        if inner:
            sections.append(("repeat", inner))
        segment_start = close + 1
        index = close + 1
    tail = text[segment_start:].strip()
    if tail:
        sections.append(("segment", tail))
    if not sections and text.strip():
        sections.append(("segment", text.strip()))
    return sections


def _tokenize_bigsmiles_fragment(text: str) -> list[tuple[str, str]]:
    """Tokenize enough BigSMILES syntax for a faithful schematic, not a chemical validator."""
    tokens: list[tuple[str, str]] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char == "[":
            close = _find_matching_token(text, index, "[", "]")
            if close is None:
                tokens.append(("unknown", text[index:]))
                break
            raw = text[index : close + 1]
            content = raw[1:-1].strip()
            if content.startswith(("$", "<", ">")) or content in {"*", "?"}:
                tokens.append(("connector", raw))
            else:
                tokens.append(("atom", raw))
            index = close + 1
            continue
        if char in _BIGSMILES_BOND_SYMBOLS:
            tokens.append(("bond", char))
            index += 1
            continue
        if char in "()":
            tokens.append(("branch_open" if char == "(" else "branch_close", char))
            index += 1
            continue
        if char == ".":
            tokens.append(("dot", char))
            index += 1
            continue
        if char in ",|;":
            tokens.append(("separator", char))
            index += 1
            continue
        if char == "%":
            match = re.match(r"%\d{2,3}", text[index:])
            if match:
                tokens.append(("ring", match.group(0)))
                index += len(match.group(0))
                continue
        if char.isdigit():
            match = re.match(r"\d+", text[index:])
            assert match is not None
            tokens.append(("ring", match.group(0)))
            index += len(match.group(0))
            continue
        if text[index : index + 2] in _BIGSMILES_TWO_LETTER_ATOMS:
            tokens.append(("atom", text[index : index + 2]))
            index += 2
            continue
        if char in _BIGSMILES_ORGANIC_ATOMS:
            tokens.append(("atom", char))
            index += 1
            continue
        if char in "{}":
            tokens.append(("unknown", char))
            index += 1
            continue
        if char in "$<>*":
            tokens.append(("connector", char))
            index += 1
            continue
        # Keep an unfamiliar token visible rather than dropping it silently.
        end = index + 1
        while end < len(text) and text[end] not in "[](){}.-=#:~/\\,.|;0123456789$<>*":
            end += 1
        tokens.append(("unknown", text[index:end]))
        index = end
    return tokens


def _build_schematic_graph(text: str) -> _SchematicGraph:
    nodes: list[_SchematicNode] = []
    edges: list[_SchematicEdge] = []
    annotations: list[tuple[str, float, float]] = []
    warnings: list[str] = []
    current: int | None = None
    cursor_x = 0.0
    cursor_y = 0.0
    pending_bond = "-"
    ring_open: dict[str, tuple[int, str]] = {}
    # parent index, main-chain restore position
    branch_stack: list[tuple[int | None, float, float]] = []

    for kind, token in _tokenize_bigsmiles_fragment(text):
        if kind == "bond":
            pending_bond = token
            continue
        if kind == "branch_open":
            if current is None:
                warnings.append("发现没有前置原子的分支")
                continue
            branch_stack.append((current, nodes[current].x + 1.0, nodes[current].y))
            direction = -1.0 if len(branch_stack) % 2 else 1.0
            cursor_x = nodes[current].x + 0.62
            cursor_y = nodes[current].y + direction * 1.0
            continue
        if kind == "branch_close":
            if not branch_stack:
                warnings.append("发现没有对应左括号的右括号")
                continue
            parent, cursor_x, cursor_y = branch_stack.pop()
            current = parent
            pending_bond = "-"
            continue
        if kind in {"dot", "separator"}:
            if kind == "separator":
                annotations.append((token, cursor_x, cursor_y))
            current = None
            cursor_x = 0.0
            cursor_y += 2.15
            pending_bond = "-"
            continue
        if kind == "ring":
            if current is None:
                warnings.append(f"环编号 {token} 没有连接到原子")
                continue
            if token in ring_open:
                start, start_bond = ring_open.pop(token)
                edges.append(_SchematicEdge(start, current, pending_bond if pending_bond != "-" else start_bond))
                nodes[start].ring_labels.append(token)
                nodes[current].ring_labels.append(token)
            else:
                ring_open[token] = (current, pending_bond)
                nodes[current].ring_labels.append(token)
            pending_bond = "-"
            continue
        if kind not in {"atom", "connector", "unknown"}:
            continue
        node = _SchematicNode(token, kind, cursor_x, cursor_y, [])
        nodes.append(node)
        new_index = len(nodes) - 1
        if current is not None:
            edges.append(_SchematicEdge(current, new_index, pending_bond))
        current = new_index
        cursor_x += 1.0
        pending_bond = "-"

    if ring_open:
        warnings.append("存在未闭合的环编号：" + ", ".join(sorted(ring_open)))
    return _SchematicGraph(nodes, edges, annotations, warnings)


def _graph_bounds(graph: _SchematicGraph) -> tuple[float, float, float, float]:
    if not graph.nodes:
        return 0.0, 1.0, 0.0, 1.0
    xs = [node.x for node in graph.nodes]
    ys = [node.y for node in graph.nodes]
    max_label = max((len(node.text) for node in graph.nodes), default=1)
    return min(xs), max(xs) + max(1.5, max_label * 0.12), min(ys) - 0.8, max(ys) + 0.8


def _load_schematic_font(size: int, bold: bool = False):
    if ImageFont is None:
        return None
    candidates = (
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, max(9, int(size)))
            except Exception:
                continue
    return ImageFont.load_default()


def _text_size(draw, text: str, font) -> tuple[int, int]:
    try:
        box = draw.textbbox((0, 0), text, font=font)
        return max(1, box[2] - box[0]), max(1, box[3] - box[1])
    except Exception:
        return max(1, len(text) * 8), 14


def _node_palette(node: _SchematicNode) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if node.kind == "connector":
        return (224, 246, 242), (0, 121, 107)
    if node.kind == "unknown":
        return (245, 238, 255), (111, 63, 160)
    label = node.text.strip("[]")
    element = re.match(r"([A-Z][a-z]?|[bcnops])", label)
    symbol = element.group(1) if element else "C"
    colors = {
        "N": ((225, 238, 255), (30, 90, 170)),
        "O": ((255, 231, 231), (180, 40, 40)),
        "S": ((255, 245, 200), (165, 112, 0)),
        "P": ((255, 235, 210), (175, 88, 0)),
        "F": ((226, 247, 226), (30, 130, 55)),
        "Cl": ((226, 247, 226), (30, 130, 55)),
        "Br": ((255, 232, 220), (145, 70, 20)),
        "I": ((239, 229, 248), (105, 50, 145)),
    }
    return colors.get(symbol, ((250, 250, 250), (70, 70, 70)))


def _draw_schematic_bond(draw, start: tuple[float, float], end: tuple[float, float], bond: str, width: int) -> None:
    x1, y1 = start
    x2, y2 = end
    line_width = max(1, width)
    if bond in {"=", "#"}:
        dx, dy = x2 - x1, y2 - y1
        length = max(math.hypot(dx, dy), 1.0)
        ox, oy = -dy / length * (3 + line_width), dx / length * (3 + line_width)
        draw.line((x1 + ox, y1 + oy, x2 + ox, y2 + oy), fill=(60, 60, 60), width=line_width)
        draw.line((x1 - ox, y1 - oy, x2 - ox, y2 - oy), fill=(60, 60, 60), width=line_width)
        if bond == "#":
            draw.line((x1, y1, x2, y2), fill=(60, 60, 60), width=line_width)
        return
    if bond == ":":
        length = max(math.hypot(x2 - x1, y2 - y1), 1.0)
        count = max(3, int(length / 8))
        for index in range(count):
            fraction = index / count
            next_fraction = min(1.0, fraction + 0.45 / count)
            draw.line(
                (x1 + (x2 - x1) * fraction, y1 + (y2 - y1) * fraction,
                 x1 + (x2 - x1) * next_fraction, y1 + (y2 - y1) * next_fraction),
                fill=(60, 60, 60), width=line_width,
            )
        return
    draw.line((x1, y1, x2, y2), fill=(70, 70, 70), width=line_width)


def _draw_schematic_graph(
    draw,
    graph: _SchematicGraph,
    origin_x: float,
    origin_y: float,
    unit_px: float,
    font,
) -> None:
    min_x, _, min_y, _ = _graph_bounds(graph)
    positions: list[tuple[float, float]] = []
    for node in graph.nodes:
        positions.append((origin_x + (node.x - min_x) * unit_px, origin_y + (node.y - min_y) * unit_px))
    for edge in graph.edges:
        _draw_schematic_bond(draw, positions[edge.start], positions[edge.end], edge.bond, max(1, int(unit_px / 16)))
    for node, (x, y) in zip(graph.nodes, positions):
        fill, outline = _node_palette(node)
        label = node.text
        text_w, text_h = _text_size(draw, label, font)
        pad_x = max(7, int(unit_px * 0.12))
        pad_y = max(4, int(unit_px * 0.08))
        box_w = max(int(unit_px * 0.44), text_w + pad_x * 2)
        box_h = max(int(unit_px * 0.38), text_h + pad_y * 2)
        left, top = x - box_w / 2, y - box_h / 2
        if node.kind == "connector":
            draw.rounded_rectangle((left, top, left + box_w, top + box_h), radius=max(5, int(unit_px * 0.1)), fill=fill, outline=outline, width=max(1, int(unit_px / 18)))
        elif node.kind == "unknown":
            draw.rounded_rectangle((left, top, left + box_w, top + box_h), radius=max(4, int(unit_px * 0.08)), fill=fill, outline=outline, width=max(1, int(unit_px / 18)))
        else:
            draw.ellipse((left, top, left + box_w, top + box_h), fill=fill, outline=outline, width=max(1, int(unit_px / 18)))
        draw.text((x - text_w / 2, y - text_h / 2 - 1), label, fill=outline, font=font)
        if node.ring_labels:
            ring = ",".join(node.ring_labels)
            ring_font = _load_schematic_font(max(10, int(unit_px * 0.18)))
            rw, rh = _text_size(draw, ring, ring_font)
            draw.text((x + box_w / 2 - rw / 2, y - box_h / 2 - rh), ring, fill=(80, 80, 80), font=ring_font)
    for label, x, y in graph.annotations:
        annotation_font = _load_schematic_font(max(11, int(unit_px * 0.22)))
        px = origin_x + (x - min_x) * unit_px
        py = origin_y + (y - min_y) * unit_px + unit_px * 0.55
        draw.text((px, py), label, fill=(100, 100, 100), font=annotation_font)


def _render_bigsmiles_algorithmic(
    raw: str,
    output_path: Path,
    options: RenderOptions,
) -> tuple[list[tuple[str, str, _SchematicGraph]], list[str]]:
    if Image is None or ImageDraw is None:
        raise RuntimeError("当前 Python 环境未安装 Pillow，无法使用 BigSMILES 算法绘图器")
    sections = []
    warnings: list[str] = []
    for kind, content in _split_bigsmiles_sections(raw):
        graph = _build_schematic_graph(content)
        sections.append((kind, content, graph))
        warnings.extend(graph.warnings)
    if not sections:
        raise ValueError("没有可绘制的 BigSMILES 片段")

    image = Image.new("RGB", (int(options.image_width), int(options.image_height)), "white")
    draw = ImageDraw.Draw(image)
    title_font = _load_schematic_font(25, bold=True)
    label_font = _load_schematic_font(16, bold=True)
    atom_font = _load_schematic_font(18)
    note_font = _load_schematic_font(13)
    draw.text((32, 18), "BigSMILES 算法结构示意图", fill=(28, 45, 60), font=title_font)
    draw.text((32, 54), "直接依据原子、键、分支、环编号和连接端布局；不转化为普通 SMILES。", fill=(80, 90, 100), font=note_font)

    extents = []
    for kind, content, graph in sections:
        min_x, max_x, min_y, max_y = _graph_bounds(graph)
        max_label = max((len(node.text) for node in graph.nodes), default=1)
        width_units = max(max_x - min_x + 1.2, 2.8 + max_label * 0.10)
        height_units = max_y - min_y + 1.8
        extents.append((width_units, height_units, min_x, min_y, max_x, max_y))
    gap_units = 1.35
    total_units = sum(item[0] for item in extents) + gap_units * max(0, len(extents) - 1)
    max_height_units = max(item[1] for item in extents)
    available_w = max(220, options.image_width - 64)
    available_h = max(180, options.image_height - 142)
    unit_px = min(82.0, available_w / max(total_units, 1.0), available_h / max(max_height_units, 1.0))
    unit_px = max(24.0, unit_px)
    if unit_px < 38:
        atom_font = _load_schematic_font(max(10, int(unit_px * 0.36)))
        label_font = _load_schematic_font(max(10, int(unit_px * 0.30)), bold=True)
    content_height = max_height_units * unit_px
    top = 108 + max(0.0, (available_h - content_height) / 2)
    x = 32.0
    for index, ((kind, content, graph), extent) in enumerate(zip(sections, extents)):
        width_units, height_units, min_x, min_y, _, _ = extent
        panel_width = width_units * unit_px
        panel_height = height_units * unit_px
        panel_top = top + (content_height - panel_height) / 2
        if kind == "repeat":
            draw.rounded_rectangle(
                (x, panel_top - 26, x + panel_width, panel_top + panel_height + 22),
                radius=12,
                fill=(255, 250, 229),
                outline=(0, 121, 107),
                width=max(2, int(unit_px / 24)),
            )
            panel_label = "重复单元 / 随机对象"
            label_color = (0, 105, 92)
        else:
            panel_label = "链外片段"
            label_color = (70, 80, 90)
        draw.text((x + 8, panel_top - 22), panel_label, fill=label_color, font=label_font)
        _draw_schematic_graph(draw, graph, x + 20, panel_top + 18, unit_px, atom_font)
        if index < len(sections) - 1:
            arrow_x1 = x + panel_width + 8
            arrow_x2 = arrow_x1 + gap_units * unit_px - 16
            arrow_y = panel_top + panel_height / 2
            draw.line((arrow_x1, arrow_y, arrow_x2, arrow_y), fill=(130, 145, 155), width=max(1, int(unit_px / 18)))
            draw.polygon(((arrow_x2, arrow_y), (arrow_x2 - 8, arrow_y - 5), (arrow_x2 - 8, arrow_y + 5)), fill=(130, 145, 155))
        x += panel_width + gap_units * unit_px

    raw_note = raw.strip().replace("\n", " ")
    if len(raw_note) > 150:
        raw_note = raw_note[:147] + "..."
    draw.text((32, options.image_height - 42), f"原始表达：{raw_note}", fill=(90, 90, 90), font=note_font)
    image.save(output_path, format="PNG")
    return sections, warnings


def _render_bigsmiles_sample_algorithmic(
    sections: list[tuple[str, str, _SchematicGraph]],
    output_path: Path,
    options: RenderOptions,
) -> bool:
    if Image is None or ImageDraw is None:
        raise RuntimeError("当前 Python 环境未安装 Pillow，无法生成采样链段图")
    repeat = next((item for item in sections if item[0] == "repeat" and item[2].nodes), None)
    if repeat is None:
        return False
    graph = repeat[2]
    min_x, max_x, min_y, max_y = _graph_bounds(graph)
    extent_w = max_x - min_x + 1.2
    extent_h = max_y - min_y + 1.8
    count = int(options.repeat_units)
    visible = count if count <= 5 else 3
    omitted = count - visible if count > 5 else 0
    panel_gap = 0.55
    ellipsis_units = 2.55 if omitted else 0.0
    total_units = extent_w * visible + panel_gap * visible + ellipsis_units
    unit_px = min(74.0, (options.image_width - 64) / max(total_units, 1), (options.image_height - 150) / max(extent_h + 1.2, 1))
    unit_px = max(24.0, unit_px)
    image = Image.new("RGB", (int(options.image_width), int(options.image_height)), "white")
    draw = ImageDraw.Draw(image)
    title_font = _load_schematic_font(24, bold=True)
    label_font = _load_schematic_font(15, bold=True)
    atom_font = _load_schematic_font(max(10, int(unit_px * 0.35)))
    note_font = _load_schematic_font(13)
    draw.text((32, 18), f"代表性采样链段示意图（重复单元数：{count}）", fill=(28, 45, 60), font=title_font)
    draw.text((32, 54), "仅展示重复单元的算法复制结果，不表示真实链长、分子量或唯一微观结构。", fill=(80, 90, 100), font=note_font)
    panel_h = extent_h * unit_px
    top = 118 + max(0, (options.image_height - 150 - panel_h) / 2)
    x = 32.0
    blocks = list(range(visible))
    if omitted:
        blocks = [0, 1, 2]
    for block_index, _ in enumerate(blocks):
        panel_w = extent_w * unit_px
        draw.rounded_rectangle(
            (x, top - 24, x + panel_w, top + panel_h + 20),
            radius=10,
            fill=(255, 250, 229),
            outline=(0, 121, 107),
            width=max(2, int(unit_px / 24)),
        )
        draw.text((x + 8, top - 20), f"重复单元 {block_index + 1}", fill=(0, 105, 92), font=label_font)
        _draw_schematic_graph(draw, graph, x + 18, top + 17, unit_px, atom_font)
        x += panel_w
        draw.line((x + 4, top + panel_h / 2, x + panel_gap * unit_px - 8, top + panel_h / 2), fill=(130, 145, 155), width=max(1, int(unit_px / 18)))
        draw.polygon(((x + panel_gap * unit_px - 8, top + panel_h / 2), (x + panel_gap * unit_px - 16, top + panel_h / 2 - 5), (x + panel_gap * unit_px - 16, top + panel_h / 2 + 5)), fill=(130, 145, 155))
        x += panel_gap * unit_px

    if omitted:
        ellipsis_width = ellipsis_units * unit_px
        draw.rounded_rectangle(
            (x, top - 24, x + ellipsis_width, top + panel_h + 20),
            radius=10,
            fill=(248, 250, 252),
            outline=(150, 160, 170),
            width=max(1, int(unit_px / 24)),
        )
        ellipsis_font = _load_schematic_font(max(15, int(unit_px * 0.34)), bold=True)
        label = f"… × {omitted} …"
        label_w, label_h = _text_size(draw, label, ellipsis_font)
        draw.text(
            (x + (ellipsis_width - label_w) / 2, top + (panel_h - label_h) / 2),
            label,
            fill=(80, 95, 110),
            font=ellipsis_font,
        )
    draw.text((32, options.image_height - 42), "连接端保留为 BigSMILES 标记，图像仅用于结构检查和沟通。", fill=(90, 90, 90), font=note_font)
    image.save(output_path, format="PNG")
    return True


#: 第三方解析库异常在界面上的展示上限（字符）。
_LIBRARY_ERROR_LIMIT = 200


def _summarize_library_error(message: object) -> str:
    """把第三方 BigSMILES 解析库的异常压成一句可直接阅读的人话。

    Olsen Lab 的 ``bigsmiles`` 库失败时会把「整段原始输入回显 + 逐 token dump」塞进
    异常信息（``Parsing failed on '{...}'.\n\tIssue with token ...``）：输入稍长就会把
    界面刷成一屏乱码。这里剥掉输入回显、折叠换行，并限制长度。
    """
    text = re.sub(r"\s+", " ", str(message or "")).strip()
    if not text:
        return ""
    marker = "Parsing failed on '"
    start = text.find(marker)
    if start >= 0:
        end = text.find("'", start + len(marker))
        if end >= 0:
            reason = text[end + 1 :].lstrip(" .")
            if reason:
                text = reason
    if len(text) > _LIBRARY_ERROR_LIMIT:
        text = text[: _LIBRARY_ERROR_LIMIT - 1].rstrip() + "…"
    return text


def _diagnose_bigsmiles_syntax(text: str) -> list[str]:
    """给出「照着改就能过」的语法建议，而不是把解析库的原始异常丢给用户。"""
    hints: list[str] = []
    compact = re.sub(r"\s+", "", text)

    empty_brackets = compact.count("[]")
    if empty_brackets:
        hints.append(
            f"检测到 {empty_brackets} 处空方括号 []：方括号内必须写内容，"
            "BigSMILES 连接端应写成 [<]、[>] 或 [$]，dummy 原子用 [*]"
        )

    missing_end_group = False
    index = 0
    while index < len(compact):
        if compact[index] != "{":
            index += 1
            continue
        close = _find_matching_token(compact, index, "{", "}")
        if close is None:
            break
        for alternative in _split_bigsmiles_alternatives(compact[index + 1 : close]):
            if not _CONNECTOR_TOKEN_RE.match(alternative):
                missing_end_group = True
                break
        index = close + 1
    if missing_end_group:
        hints.append(
            "随机对象 {…} 没有声明端基：官方规范要求写成 {[>][<]重复单元[>][<]}，"
            "只写 {重复单元} 会被判定为缺少端基而无法验证"
        )
    return hints


def _try_package_parse(text: str) -> tuple[bool, str, str, str, str]:
    """调用 Olsen Lab 官方 BigSMILES 解析器。

    返回 ``(是否通过, 规范化结构, 渲染器描述, 错误信息, 解析器状态)``；状态取
    ``accepted``（通过）/ ``rejected``（入口存在但拒绝了该表达式）/ ``unavailable``
    （未安装或没有可调用入口）。
    """
    try:
        module = importlib.import_module("bigsmiles")
    except Exception as exc:
        return (
            False,
            text,
            "BigSMILES 保守检查（未安装 Olsen Lab bigsmiles）",
            f"未安装 Olsen Lab BigSMILES 解析库：{_summarize_library_error(exc)}",
            "unavailable",
        )
    version = getattr(module, "__version__", "未知版本")
    rejected_renderer = f"BigSMILES 保守检查（Olsen Lab bigsmiles {version} 未通过）"
    parser_names = ("BigSMILES", "parse", "parse_bigsmiles", "from_string", "from_bigsmiles")
    last_error = ""
    called = False
    for name in parser_names:
        parser = getattr(module, name, None)
        if not callable(parser):
            continue
        called = True
        try:
            parsed = parser(text)
        except Exception as exc:
            last_error = _summarize_library_error(exc)
            continue
        normalized = str(parsed).strip() or text
        return True, normalized, f"Olsen Lab bigsmiles {version}", "", "accepted"
    if called:
        # 关键区分：入口存在、也真的执行了，是它「拒绝」了这条表达式；
        # 不能写成「未找到可用入口」，那会把用户引到错误的方向。
        message = "官方解析器拒绝该表达式"
        if last_error:
            message += f"：{last_error}"
        return False, text, rejected_renderer, message, "rejected"
    return (
        False,
        text,
        f"BigSMILES 保守检查（Olsen Lab bigsmiles {version} 无可调用入口）",
        "bigsmiles 库中没有找到可调用的解析入口",
        "unavailable",
    )


def _parse_bigsmiles(text: str) -> _BigSMILESParse:
    package_ok, normalized, package_renderer, package_error, parser_status = _try_package_parse(text)
    if package_ok:
        return _BigSMILESParse(
            valid=True,
            normalized=normalized,
            fragments=_candidate_fragments(text),
            renderer=package_renderer,
            parser_status=parser_status,
        )
    hints = _diagnose_bigsmiles_syntax(text)
    if not _balanced_bigsmiles(text):
        return _BigSMILESParse(
            valid=False,
            normalized=text,
            fragments=[],
            renderer=package_renderer,
            parser_status=parser_status,
            error="；".join(["BigSMILES 括号、方括号或随机对象边界不匹配", *hints]),
        )
    fragments = _candidate_fragments(text)
    if not fragments:
        return _BigSMILESParse(
            valid=False,
            normalized=text,
            fragments=[],
            renderer=package_renderer,
            parser_status=parser_status,
            error="；".join(["未找到可识别的 BigSMILES 重复单元或连接片段", *hints]),
        )
    warning = "；".join(
        value for value in [
            "官方 BigSMILES 解析器未接受该表达式，已回退到本平台的保守语法检查，结果仅供结构核对",
            package_error,
            *hints,
        ] if value
    )
    return _BigSMILESParse(
        valid=True,
        normalized=text,
        fragments=fragments,
        renderer=package_renderer,
        parser_status=parser_status,
        warning=warning,
    )


def _fragment_molecules(fragments: list[str]) -> list:
    if Chem is None:
        return []
    molecules = []
    for fragment in fragments:
        cleaned = fragment.replace("[*]", "")
        if not cleaned:
            continue
        try:
            molecule = Chem.MolFromSmiles(cleaned)
        except Exception:
            molecule = None
        if molecule is not None:
            molecules.append(molecule)
    return molecules



@dataclass
class _BigSMILESFragmentSpec:
    """一个独立绘图片段；RDKit 只负责片段内部的化学二维布局。"""

    kind: str
    raw: str
    molecule: object | None
    connectors: list[str]
    parse_mode: str
    warning: str = ""
    #: 组件归属标签（多组分输入时形如“组分 2/3”，单组分为空）。
    component: str = ""
    #: 需要在图中高亮的原子下标（用于“重复单元已展开”的完整骨架）。
    highlight_atoms: tuple[int, ...] = ()
    #: 展开时随机对象含多个候选，图中仅取了第一个候选。
    has_alternatives: bool = False


_CONNECTOR_TOKEN_RE = re.compile(r"\[\s*(?:(?:[$<>][^\]]*)|[?*])\s*\]")


def _split_bigsmiles_alternatives(text: str) -> list[str]:
    """按顶层随机对象分隔符拆分，不拆括号、方括号和环结构内部。"""
    values: list[str] = []
    start = 0
    stack: list[str] = []
    pairs = {"[": "]", "(": ")"}
    separators = {",", "|", ";"}
    for index, char in enumerate(text):
        if char in pairs:
            stack.append(char)
        elif char in pairs.values() and stack:
            if pairs[stack[-1]] == char:
                stack.pop()
        elif char in separators and not stack:
            value = text[start:index].strip()
            if value:
                values.append(value)
            start = index + 1
    value = text[start:].strip()
    if value:
        values.append(value)
    return values or ([text.strip()] if text.strip() else [])


def _connector_tokens(text: str) -> list[str]:
    return [match.group(0).strip() for match in _CONNECTOR_TOKEN_RE.finditer(text)]


def _replace_stochastic_objects_with_placeholders(text: str) -> tuple[str, list[str]]:
    """把嵌入式随机对象替换成 dummy 原子，同时保留其原文。"""
    values: list[str] = []
    output: list[str] = []
    index = 0
    placeholder_index = 900
    while index < len(text):
        if text[index] != "{":
            output.append(text[index])
            index += 1
            continue
        close = _find_matching_token(text, index, "{", "}")
        if close is None:
            return text, []
        values.append(text[index + 1 : close].strip())
        output.append(f"[*:{placeholder_index}]")
        placeholder_index += 1
        index = close + 1
    return "".join(output), values


def _set_placeholder_atom_labels(molecule: object, count: int) -> None:
    """为完整骨架中的随机对象占位 dummy 原子设置短标签。"""
    if molecule is None or count <= 0:
        return
    for atom in molecule.GetAtoms():
        atom_map = atom.GetAtomMapNum()
        if 900 <= atom_map < 900 + count:
            atom.SetProp("atomLabel", f"{{R{atom_map - 899}}}")


def _prepare_rdkit_bigsmiles_fragment(raw: str) -> tuple[object | None, list[str], str, str]:
    """将连接端暂时映射为 dummy atom，化学片段本身仍由 RDKit 解析。"""
    if Chem is None:
        return None, _connector_tokens(raw), "unavailable", "当前 Python 环境未安装 RDKit"
    cleaned = re.sub(r"\s+", "", raw).replace("{", "").replace("}", "")
    cleaned = cleaned.strip(".")
    connectors: list[str] = []

    def replace_connector(match: re.Match[str]) -> str:
        token = match.group(0).strip()
        connectors.append(token)
        return f"[*:{len(connectors)}]"

    rdkit_text = _CONNECTOR_TOKEN_RE.sub(replace_connector, cleaned)
    molecule = None
    try:
        molecule = Chem.MolFromSmiles(rdkit_text)
    except Exception:
        molecule = None
    if molecule is not None:
        for atom in molecule.GetAtoms():
            atom_map = atom.GetAtomMapNum()
            if atom_map and 1 <= atom_map <= len(connectors):
                atom.SetProp("atomLabel", connectors[atom_map - 1])
        return molecule, connectors, "dummy_connectors", ""

    # 某些 BigSMILES 外部骨架不是严格意义上的单个标准 SMILES：
    # 例如环氧/聚醚连接模式可能让 RDKit 的严格价态检查失败，但拓扑仍
    # 足够明确，可以用于二维结构检查。仅在严格解析失败时保留未清洗分子。
    try:
        molecule = Chem.MolFromSmiles(rdkit_text, sanitize=False)
    except Exception:
        molecule = None
    if molecule is not None:
        for atom in molecule.GetAtoms():
            atom_map = atom.GetAtomMapNum()
            if atom_map and 1 <= atom_map <= len(connectors):
                atom.SetProp("atomLabel", connectors[atom_map - 1])
        return (
            molecule,
            connectors,
            "unsanitized_topology",
            "RDKit 严格价态校验未通过；已保留拓扑用于二维绘图，不能视为已验证标准 SMILES",
        )

    # 某些连接端组合会造成 dummy atom 的价态无法校验。此时保留化学核心，
    # 连接端仍在面板标签中显示，避免把整张图退化成按字符绘制。
    core = _remove_bigsmiles_bracket_token(cleaned)
    core = re.sub(r"[{}]", "", core).strip(".")
    try:
        molecule = Chem.MolFromSmiles(core)
    except Exception:
        molecule = None
    if molecule is not None:
        return (
            molecule,
            connectors,
            "core_only",
            "连接端未能作为 RDKit dummy atom 接入，已保留化学核心并在图中单独标注连接端",
        )
    try:
        molecule = Chem.MolFromSmiles(core, sanitize=False)
    except Exception:
        molecule = None
    if molecule is not None:
        return (
            molecule,
            connectors,
            "unsanitized_core",
            "RDKit 严格价态校验未通过；已保留化学核心用于二维绘图，不能视为已验证标准 SMILES",
        )
    return None, connectors, "unparsed", f"片段无法转换为 RDKit 可解析结构：{raw}"


def _build_rdkit_bigsmiles_specs(raw: str) -> list[_BigSMILESFragmentSpec]:
    placeholder_smiles, stochastic_objects = _replace_stochastic_objects_with_placeholders(raw)
    has_embedded_stochastic_object = bool(stochastic_objects) and placeholder_smiles != raw
    specs: list[_BigSMILESFragmentSpec] = []

    # 先解析完整外部骨架。这样随机对象位于支链、酯基或环结构内部时，
    # 前后化学上下文不会被截断成两个非法 SMILES 片段。
    if has_embedded_stochastic_object:
        molecule, connectors, parse_mode, warning = _prepare_rdkit_bigsmiles_fragment(placeholder_smiles)
        _set_placeholder_atom_labels(molecule, len(stochastic_objects))
        placeholder_warning = (
            "完整外部骨架中的 BigSMILES 随机对象已用占位符表示，"
            "随机对象候选结构见独立面板"
        )
        if warning:
            placeholder_warning += "；" + warning
        specs.append(
            _BigSMILESFragmentSpec(
                kind="context",
                raw=placeholder_smiles,
                molecule=molecule,
                connectors=connectors,
                parse_mode=(
                    parse_mode
                    if parse_mode in {"unsanitized_topology", "unsanitized_core"}
                    else "context_placeholder" if molecule is not None else parse_mode
                ),
                warning=placeholder_warning if molecule is not None else warning,
            )
        )

    # 随机对象本体仍按候选项分别绘制，连接端交给 RDKit dummy atom 处理。
    for kind, content in _split_bigsmiles_sections(raw):
        alternatives = _split_bigsmiles_alternatives(content) if kind == "repeat" else [content]
        # 嵌入式随机对象已经由完整 context spec 表示；不要再添加被大括号
        # 截断的 segment，否则会重新出现“该片段无法由 RDKit 解析”。
        if has_embedded_stochastic_object and kind == "segment":
            continue
        for alternative in alternatives:
            molecule, connectors, parse_mode, warning = _prepare_rdkit_bigsmiles_fragment(alternative)
            specs.append(
                _BigSMILESFragmentSpec(
                    kind=kind,
                    raw=alternative,
                    molecule=molecule,
                    connectors=connectors,
                    parse_mode=parse_mode,
                    warning=warning,
                )
            )
    return specs


_BIGSMILES_PANEL_STYLE: dict[str, dict[str, object]] = {
    "expanded": {
        "title": "完整骨架（重复单元已展开）",
        "fill": (236, 248, 243),
        "outline": (0, 121, 107),
        "title_color": (0, 105, 92),
    },
    "context": {
        "title": "完整外部骨架（随机对象占位）",
        "fill": (246, 249, 253),
        "outline": (83, 116, 150),
        "title_color": (45, 82, 125),
    },
    "repeat": {
        "title": "重复单元 / 随机对象",
        "fill": (255, 250, 229),
        "outline": (0, 121, 107),
        "title_color": (0, 105, 92),
    },
    "segment": {
        "title": "链外片段",
        "fill": (248, 250, 252),
        "outline": (150, 160, 170),
        "title_color": (70, 80, 90),
    },
}


def _spec_note(spec: _BigSMILESFragmentSpec) -> str:
    if spec.kind == "expanded":
        suffix = "（随机对象含多个候选，图中取第一个候选）" if spec.has_alternatives else ""
        if spec.highlight_atoms:
            return "高亮部分为重复单元原子；已按连接端就地展开成连通骨架" + suffix
        return "已把 {重复单元} 就地展开为连通的完整骨架" + suffix
    if spec.kind == "context":
        maps = re.findall(r"\[\*:(\d+)\]", spec.raw)
        labels = " ".join(f"{{R{int(value) - 899}}}" for value in maps)
        return "随机对象占位：" + (labels or "{R}")
    if not spec.connectors:
        return "连接端：无"
    return "连接端：" + " ".join(spec.connectors)


def _panel_for_spec(spec: _BigSMILESFragmentSpec) -> _Panel:
    style = _BIGSMILES_PANEL_STYLE.get(spec.kind, _BIGSMILES_PANEL_STYLE["segment"])
    title = str(style["title"])
    if spec.component:
        title = f"{spec.component} · {title}"
    return _Panel(
        title=title,
        note=_spec_note(spec),
        molecule=spec.molecule,
        highlight_atoms=tuple(spec.highlight_atoms),
        fill=style["fill"],
        outline=style["outline"],
        title_color=style["title_color"],
    )


def _strip_bigsmiles_connectors(text: str) -> str:
    """删除 BigSMILES 连接端标记，保留原子、键与环信息的纯 SMILES 文本。"""
    cleaned = _CONNECTOR_TOKEN_RE.sub("", text)
    cleaned = cleaned.replace("{", "").replace("}", "")
    return re.sub(r"\s+", "", cleaned)


def _expand_repeat_units(text: str) -> tuple[str, list[tuple[int, int]]]:
    """把 ``{...}`` 就地展开成完整骨架文本。

    返回 ``(展开后的 SMILES, [(插入片段起始字符下标, 长度)])``。BigSMILES 的连接端
    ``[>] [<] [$…]`` 本身只是键端声明，去掉后剩下的原子序列就是真实的连线顺序。
    随机对象含多个候选（``a,b``）时取第一个候选，以便仍然能给出一张连通骨架。
    """
    output: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    index = 0
    while index < len(text):
        if text[index] != "{":
            output.append(text[index])
            cursor += 1
            index += 1
            continue
        close = _find_matching_token(text, index, "{", "}")
        if close is None:
            output.append(text[index])
            cursor += 1
            index += 1
            continue
        content = text[index + 1 : close]
        alternatives = _split_bigsmiles_alternatives(content)
        inserted = _strip_bigsmiles_connectors(alternatives[0] if alternatives else content)
        if inserted:
            spans.append((cursor, len(inserted)))
            output.append(inserted)
            cursor += len(inserted)
        index = close + 1
    return "".join(output), spans


def _atom_offsets(text: str) -> list[int]:
    """每个原子 token 的起始字符下标（RDKit 按文本顺序分配原子下标）。"""
    return [match.start() for match in _SMILES_ATOM_TOKEN_RE.finditer(text)]


def _has_repeat_alternatives(text: str) -> bool:
    """随机对象是否写成多候选形式（``a,b``）。"""
    index = 0
    while index < len(text):
        if text[index] != "{":
            index += 1
            continue
        close = _find_matching_token(text, index, "{", "}")
        if close is None:
            return False
        if len(_split_bigsmiles_alternatives(text[index + 1 : close])) > 1:
            return True
        index = close + 1
    return False


def _quiet_smiles_parse(text: str, sanitize: bool):
    """尝试性解析（不污染日志）：失败时返回 None。"""
    if Chem is None:
        return None
    try:
        from rdkit import rdBase

        with rdBase.BlockLogs():
            return Chem.MolFromSmiles(text, sanitize=sanitize)
    except Exception:
        pass
    try:
        return Chem.MolFromSmiles(text, sanitize=sanitize)
    except Exception:
        return None


def _build_expanded_framework_spec(raw: str) -> _BigSMILESFragmentSpec | None:
    """构造“重复单元已展开”的完整骨架片段（含高亮原子下标）。"""
    if Chem is None or "{" not in raw:
        return None
    expanded, spans = _expand_repeat_units(raw)
    if not expanded.strip():
        return None
    molecule = None
    for sanitize in (True, False):
        molecule = _quiet_smiles_parse(expanded, sanitize)
        if molecule is not None:
            break
    if molecule is None:
        return None

    highlight: list[int] = []
    offsets = _atom_offsets(expanded)
    if spans and len(offsets) == molecule.GetNumAtoms():
        for start, length in spans:
            end = start + length
            highlight.extend(
                index for index, offset in enumerate(offsets) if start <= offset < end
            )
    return _BigSMILESFragmentSpec(
        kind="expanded",
        raw=expanded,
        molecule=molecule,
        connectors=_connector_tokens(raw),
        parse_mode="expanded_framework",
        highlight_atoms=tuple(sorted(set(highlight))),
        has_alternatives=_has_repeat_alternatives(raw),
    )


def _bigsmiles_component_specs(raw: str, options: RenderOptions) -> list[_BigSMILESFragmentSpec]:
    """按顶层 ``.`` 拆分组件，并为每个组件前插“完整骨架”面板。"""
    components = [part for part in _split_top_level(raw, ".") if part.strip()] or [raw.strip()]
    multi = len(components) > 1
    collected: list[_BigSMILESFragmentSpec] = []
    for index, component in enumerate(components):
        label = f"组分 {index + 1}/{len(components)}" if multi else ""
        group: list[_BigSMILESFragmentSpec] = []
        if getattr(options, "expand_repeat_units", True):
            expanded = _build_expanded_framework_spec(component)
            if expanded is not None:
                group.append(expanded)
        group.extend(_build_rdkit_bigsmiles_specs(component))
        for spec in group:
            spec.component = label
        collected.extend(group)
    return collected


def _chunk_rows(items: list, per_row: int) -> list[list]:
    """把面板切成尽量均匀的若干行，避免最后一行只剩一个面板被拉得很大。"""
    total = len(items)
    if total == 0:
        return []
    per_row = max(1, int(per_row))
    rows = math.ceil(total / per_row)
    base, extra = divmod(total, rows)
    out: list[list] = []
    cursor = 0
    for index in range(rows):
        size = base + (1 if index < extra else 0)
        out.append(items[cursor:cursor + size])
        cursor += size
    return out


def _render_bigsmiles_rdkit(
    raw: str,
    output_path: Path,
    options: RenderOptions,
) -> tuple[list[_BigSMILESFragmentSpec], list[str]]:
    if Image is None or ImageDraw is None or Draw is None or Chem is None:
        raise RuntimeError("RDKit 或 Pillow 不可用")
    specs = _bigsmiles_component_specs(raw, options)
    drawable_specs = [spec for spec in specs if spec.molecule is not None]
    if not drawable_specs:
        raise ValueError("没有可由 RDKit 绘制的 BigSMILES 化学片段")

    warnings: list[str] = []
    for spec in specs:
        if spec.warning and spec.warning not in warnings:
            warnings.append(spec.warning)
    if any(spec.parse_mode == "core_only" for spec in specs):
        warnings.append("部分连接端仅作为语义标记显示，未强行伪装成完整聚合物分子")
    if any(spec.kind == "expanded" for spec in specs):
        warnings.append("已额外输出“重复单元已展开”的完整骨架图（高亮原子即重复单元部分）")

    panels = [_panel_for_spec(spec) for spec in specs]
    per_row = 3 if len(panels) >= 3 else max(1, len(panels))
    raw_note = raw.replace("\n", " ").strip()
    if len(raw_note) > 130:
        raw_note = raw_note[:127] + "..."
    _compose_panel_figure(
        _chunk_rows(panels, per_row),
        output_path,
        title="BigSMILES 片段化学结构图",
        subtitle="片段内部由 RDKit 二维布局；连接端与重复单元语义单独保留。图高按内容自适应。",
        footer=f"原始表达：{raw_note}",
        width=int(options.image_width),
        scale=_supersample_scale(options),
    )
    return specs, warnings


def _render_bigsmiles_sample_rdkit(
    specs: list[_BigSMILESFragmentSpec],
    output_path: Path,
    options: RenderOptions,
) -> bool:
    """代表性采样链段图：重复单元整体复制，并用省略号面板表示剩余重复数。"""
    repeat = next((spec for spec in specs if spec.kind == "repeat" and spec.molecule is not None), None)
    if repeat is None or Image is None or ImageDraw is None:
        return False
    count = int(options.repeat_units)
    visible = count if count <= 4 else 3
    omitted = max(0, count - visible)
    note = _spec_note(repeat)
    panels: list[_Panel] = []
    for index in range(visible):
        panels.append(
            _Panel(
                title=f"重复单元 {index + 1}" if visible > 1 else "重复单元",
                note=note,
                molecule=repeat.molecule,
                fill=(255, 250, 229),
                outline=(0, 121, 107),
                title_color=(0, 105, 92),
            )
        )
    if omitted:
        panels.append(
            _Panel(
                title="… 省略…",
                note=f"共 {count} 个重复单元，此处省略 {omitted} 个",
                molecule=None,
                fill=(248, 250, 252),
                outline=(150, 160, 170),
                title_color=(70, 80, 90),
                empty_text=f"… x {omitted} …",
                empty_color=(80, 95, 110),
            )
        )
    _compose_panel_figure(
        _chunk_rows(panels, 4),
        output_path,
        title=f"代表性采样链段示意图（重复单元数：{count}）",
        subtitle="重复单元图像由 RDKit 生成后整体复制；不表示真实链长、分子量或唯一微观结构。",
        footer="BigSMILES 连接端保留为语义标记，采样图仅用于拓扑检查。",
        width=int(options.image_width),
        scale=_supersample_scale(options),
        max_cell_h=420.0,
    )
    return True

def _render_bigsmiles(raw: str, output_dir: Path, options: RenderOptions) -> RenderResult:
    parsed = _parse_bigsmiles(raw.strip())
    disclaimers = (
        "主图表示重复单元/连接模式视图，不代表唯一完整聚合物分子",
        "二维位置用于结构检查和沟通，不代表真实聚合物构象",
    )
    if not parsed.valid:
        result = _invalid_result(raw, "bigsmiles", parsed.error, parsed.renderer)
        result.disclaimers = disclaimers
        result.parser_status = parsed.parser_status
        return result
    # 固定免责说明不再混进 warning_message，否则会把真正需要用户处理的问题淹没。
    warnings: list[str] = []
    if parsed.warning:
        warnings.append(parsed.warning)
    image_name = _hash_name(raw.strip(), "main")
    use_rdkit_fragments = Chem is not None and Draw is not None and Image is not None and ImageDraw is not None
    specs: list[_BigSMILESFragmentSpec] = []
    sections: list[tuple[str, str, _SchematicGraph]] = []
    try:
        if use_rdkit_fragments:
            specs, fragment_warnings = _render_bigsmiles_rdkit(raw.strip(), output_dir / image_name, options)
            warnings.extend(fragment_warnings)
        else:
            sections, schematic_warnings = _render_bigsmiles_algorithmic(raw.strip(), output_dir / image_name, options)
            warnings.extend(schematic_warnings)
    except Exception as exc:
        try:
            sections, schematic_warnings = _render_bigsmiles_algorithmic(raw.strip(), output_dir / image_name, options)
            warnings.extend([f"RDKit 片段布局未完全成功：{exc}", "已退回保守示意图"] + schematic_warnings)
            use_rdkit_fragments = False
        except Exception as fallback_exc:
            return RenderResult(
                raw_string=raw,
                detected_type="bigsmiles",
                parse_status="valid",
                normalized_structure=parsed.normalized,
                draw_status="render_failed",
                error_message=f"BigSMILES 绘图失败：{fallback_exc}",
                warning_message="；".join(warnings),
                renderer=f"{parsed.renderer}；BigSMILES 绘图器",
                disclaimers=disclaimers,
                parser_status=parsed.parser_status,
            )
    sample_name = ""
    draw_status: DrawStatus = "rendered"
    if options.render_sample_chain:
        sample_name = _hash_name(raw.strip(), f"sample_{int(options.repeat_units)}")
        try:
            if use_rdkit_fragments:
                sample_ok = _render_bigsmiles_sample_rdkit(specs, output_dir / sample_name, options)
            else:
                sample_ok = _render_bigsmiles_sample_algorithmic(sections, output_dir / sample_name, options)
            if not sample_ok:
                sample_name = ""
                warnings.append("表达式中没有可用于复制的重复单元，未生成代表性采样链段图")
                draw_status = "main_rendered_sample_failed"
            else:
                warnings.append("代表性采样链段图由 RDKit 片段整体复制，仅作拓扑示意")
        except Exception as exc:
            sample_name = ""
            warnings.append(f"代表性采样链段图生成失败：{exc}")
            draw_status = "main_rendered_sample_failed"
    return RenderResult(
        raw_string=raw,
        detected_type="bigsmiles",
        parse_status="valid",
        normalized_structure=parsed.normalized,
        main_image_path=image_name,
        sample_image_path=sample_name,
        draw_status=draw_status,
        warning_message="；".join(warnings),
        renderer=(f"{parsed.renderer}；RDKit BigSMILES 片段布局" if use_rdkit_fragments else f"{parsed.renderer}；BigSMILES 保守示意绘图器"),
        disclaimers=disclaimers,
        parser_status=parsed.parser_status,
    )


def render_structure(raw: str, output_dir: Path, options: RenderOptions | None = None) -> RenderResult:
    options = options or RenderOptions()
    raw_string = "" if raw is None else (raw if isinstance(raw, str) else str(raw))
    detected = identify_structure_type(raw_string, options.requested_type)
    output_dir = _ensure_output_dir(Path(output_dir))
    if not raw_string.strip():
        return RenderResult(
            raw_string=raw_string,
            detected_type="unknown",
            parse_status="empty",
            draw_status="not_requested",
            renderer="未解析",
        )
    if detected == "smiles":
        return _render_smiles(raw_string, output_dir, options)
    if detected == "bigsmiles":
        return _render_bigsmiles(raw_string, output_dir, options)
    return _invalid_result(raw_string, detected, "无法识别结构类型", "未解析")
