from __future__ import annotations

import hashlib
import importlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - 由结果对象给出缺少绘图依赖的提示
    Image = None
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


def _render_molecules(
    molecules: list,
    output_path: Path,
    options: RenderOptions,
    legends: list[str] | None = None,
) -> None:
    if not molecules:
        raise ValueError("没有可绘制的分子对象")
    if Draw is None:
        raise RuntimeError("当前 Python 环境未安装 RDKit，无法绘图")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(molecules) == 1:
        image = Draw.MolToImage(
            molecules[0],
            size=(int(options.image_width), int(options.image_height)),
            kekulize=False,
        )
    else:
        sub_size = (
            max(200, int(options.image_width / min(len(molecules), 4))),
            max(200, int(options.image_height / 2)),
        )
        image = Draw.MolsToGridImage(
            molecules,
            molsPerRow=min(len(molecules), 4),
            subImgSize=sub_size,
            legends=legends,
            useSVG=False,
        )
    image.save(output_path, format="PNG")


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
    try:
        _render_molecules([molecule], output_dir / image_name, options)
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
    return RenderResult(
        raw_string=raw,
        detected_type="smiles",
        parse_status="valid",
        normalized_structure=normalized,
        main_image_path=image_name,
        draw_status="rendered",
        renderer=renderer,
    )


@dataclass
class _BigSMILESParse:
    valid: bool
    normalized: str
    fragments: list[str]
    renderer: str
    warning: str = ""
    error: str = ""


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


def _try_package_parse(text: str) -> tuple[bool, str, str, str]:
    try:
        module = importlib.import_module("bigsmiles")
    except Exception as exc:
        return False, text, "", f"未安装 Olsen Lab BigSMILES 解析库：{exc}"
    version = getattr(module, "__version__", "未知版本")
    parser_names = ("BigSMILES", "parse", "parse_bigsmiles", "from_string", "from_bigsmiles")
    last_error = ""
    for name in parser_names:
        parser = getattr(module, name, None)
        if not callable(parser):
            continue
        try:
            parsed = parser(text)
            normalized = str(parsed).strip() or text
            return True, normalized, f"Olsen Lab bigsmiles {version}", ""
        except Exception as exc:
            last_error = str(exc)
    message = "BigSMILES 库未找到可用的解析入口"
    if last_error:
        message += f"：{last_error}"
    return False, text, f"Olsen Lab bigsmiles {version}", message


def _parse_bigsmiles(text: str) -> _BigSMILESParse:
    package_ok, normalized, package_renderer, package_error = _try_package_parse(text)
    if package_ok:
        return _BigSMILESParse(
            valid=True,
            normalized=normalized,
            fragments=_candidate_fragments(text),
            renderer=package_renderer,
            warning="",
        )
    if not _balanced_bigsmiles(text):
        return _BigSMILESParse(
            valid=False,
            normalized=text,
            fragments=[],
            renderer=package_renderer or "BigSMILES 保守检查",
            error="BigSMILES 括号、方括号或随机对象边界不匹配",
        )
    fragments = _candidate_fragments(text)
    if not fragments:
        return _BigSMILESParse(
            valid=False,
            normalized=text,
            fragments=[],
            renderer=package_renderer or "BigSMILES 保守检查",
            error="未找到可识别的 BigSMILES 重复单元或连接片段",
        )
    warning = "；".join(
        value for value in [
            "未能调用 Olsen Lab BigSMILES 解析入口，已使用保守语法检查",
            package_error,
        ] if value
    )
    return _BigSMILESParse(
        valid=True,
        normalized=text,
        fragments=fragments,
        renderer=package_renderer or "BigSMILES 保守检查",
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


def _prepare_fragment_molecule_for_draw(molecule: object) -> object:
    if Chem is None:
        return molecule
    drawable = Chem.Mol(molecule)
    if rdDepictor is not None:
        rdDepictor.Compute2DCoords(drawable, canonOrient=True)
    return drawable


def _render_rdkit_fragment_image(spec: _BigSMILESFragmentSpec, size: tuple[int, int]):
    if spec.molecule is None or Draw is None:
        return None
    molecule = _prepare_fragment_molecule_for_draw(spec.molecule)
    return Draw.MolToImage(
        molecule,
        size=(max(120, int(size[0])), max(100, int(size[1]))),
        kekulize=False,
        wedgeBonds=True,
    ).convert("RGB")


def _draw_arrow(draw, x1: float, y1: float, x2: float, y2: float) -> None:
    draw.line((x1, y1, x2, y2), fill=(105, 125, 138), width=2)
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 8.0
    left = (x2 - size * math.cos(angle - math.pi / 6), y2 - size * math.sin(angle - math.pi / 6))
    right = (x2 - size * math.cos(angle + math.pi / 6), y2 - size * math.sin(angle + math.pi / 6))
    draw.polygon(((x2, y2), left, right), fill=(105, 125, 138))


def _draw_connector_caption(draw, spec: _BigSMILESFragmentSpec, x: float, y: float, width: float, font) -> None:
    if spec.kind == "context":
        maps = re.findall(r"\[\*:(\d+)\]", spec.raw)
        labels = " ".join(f"{{R{int(value) - 899}}}" for value in maps)
        text = "随机对象占位：" + (labels or "{R}")
        color = (126, 82, 0)
    elif not spec.connectors:
        text = "连接端：无"
        color = (110, 120, 128)
    else:
        text = "连接端：" + " ".join(spec.connectors)
        color = (0, 105, 92)
    draw.text((x, y), text, fill=color, font=font)


def _render_bigsmiles_rdkit(
    raw: str,
    output_path: Path,
    options: RenderOptions,
) -> tuple[list[_BigSMILESFragmentSpec], list[str]]:
    if Image is None or ImageDraw is None or Draw is None or Chem is None:
        raise RuntimeError("RDKit 或 Pillow 不可用")
    specs = _build_rdkit_bigsmiles_specs(raw)
    drawable_specs = [spec for spec in specs if spec.molecule is not None]
    if not drawable_specs:
        raise ValueError("没有可由 RDKit 绘制的 BigSMILES 化学片段")

    warnings: list[str] = []
    for spec in specs:
        if spec.warning:
            warnings.append(spec.warning)
    if any(spec.parse_mode == "core_only" for spec in specs):
        warnings.append("部分连接端仅作为语义标记显示，未强行伪装成完整聚合物分子")

    width, height = int(options.image_width), int(options.image_height)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _load_schematic_font(25, bold=True)
    label_font = _load_schematic_font(16, bold=True)
    caption_font = _load_schematic_font(13)
    note_font = _load_schematic_font(12)
    draw.text((32, 18), "BigSMILES 片段化学结构图", fill=(28, 45, 60), font=title_font)
    draw.text((32, 54), "片段内部由 RDKit 二维布局；连接端和重复单元语义单独保留。", fill=(80, 90, 100), font=caption_font)

    margin_x = 32
    content_top = 92
    footer_h = 54
    gap_x, gap_y = 18, 18
    columns = 1 if len(specs) == 1 else min(3, len(specs))
    rows = math.ceil(len(specs) / columns)
    cell_w = max(200, (width - margin_x * 2 - gap_x * (columns - 1)) / columns)
    cell_h = max(190, (height - content_top - footer_h - gap_y * (rows - 1)) / rows)

    for index, spec in enumerate(specs):
        row, column = divmod(index, columns)
        left = margin_x + column * (cell_w + gap_x)
        top = content_top + row * (cell_h + gap_y)
        right, bottom = left + cell_w, top + cell_h
        is_repeat = spec.kind == "repeat"
        is_context = spec.kind == "context"
        fill = (255, 250, 229) if is_repeat else (246, 249, 253)
        outline = (0, 121, 107) if is_repeat else (83, 116, 150) if is_context else (150, 160, 170)
        draw.rounded_rectangle((left, top, right, bottom), radius=10, fill=fill, outline=outline, width=2)
        if is_repeat:
            label = "重复单元 / 随机对象"
            label_color = (0, 105, 92)
        elif is_context:
            label = "完整外部骨架（随机对象占位）"
            label_color = (45, 82, 125)
        else:
            label = "链外片段"
            label_color = (70, 80, 90)
        if len(specs) > 1:
            label += f"  #{index + 1}"
        draw.text((left + 12, top + 8), label, fill=label_color, font=label_font)
        if spec.molecule is not None:
            mol_image = _render_rdkit_fragment_image(
                spec,
                (int(cell_w - 28), max(120, int(cell_h - 78))),
            )
            if mol_image is not None:
                mol_x = int(left + (cell_w - mol_image.width) / 2)
                mol_y = int(top + 38 + (cell_h - 72 - mol_image.height) / 2)
                image.paste(mol_image, (mol_x, mol_y))
        else:
            draw.text((left + 16, top + cell_h / 2), "该片段无法由 RDKit 解析", fill=(170, 60, 50), font=caption_font)
        _draw_connector_caption(draw, spec, left + 12, bottom - 24, cell_w - 24, note_font)
        if column < columns - 1 and index + 1 < len(specs):
            _draw_arrow(draw, right + 3, top + cell_h / 2, right + gap_x - 3, top + cell_h / 2)

    raw_note = raw.replace("\n", " ").strip()
    if len(raw_note) > 130:
        raw_note = raw_note[:127] + "..."
    draw.text((32, height - 34), f"原始表达：{raw_note}", fill=(90, 90, 90), font=note_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return specs, warnings


def _render_bigsmiles_sample_rdkit(
    specs: list[_BigSMILESFragmentSpec],
    output_path: Path,
    options: RenderOptions,
) -> bool:
    repeat = next((spec for spec in specs if spec.kind == "repeat" and spec.molecule is not None), None)
    if repeat is None or Image is None or ImageDraw is None:
        return False
    width, height = int(options.image_width), int(options.image_height)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _load_schematic_font(24, bold=True)
    label_font = _load_schematic_font(15, bold=True)
    note_font = _load_schematic_font(12)
    caption_font = _load_schematic_font(13)
    count = int(options.repeat_units)
    visible = count if count <= 4 else 3
    omitted = max(0, count - visible)
    draw.text((32, 18), f"代表性采样链段示意图（重复单元数：{count}）", fill=(28, 45, 60), font=title_font)
    draw.text((32, 54), "重复单元图像由 RDKit 生成后整体复制；不表示真实链长、分子量或唯一微观结构。", fill=(80, 90, 100), font=caption_font)

    item_count = visible + (1 if omitted else 0)
    gap = 14
    item_w = max(170, (width - 64 - gap * (item_count - 1)) / item_count)
    top, panel_h = 125, min(430, height - 180)
    mol_image = _render_rdkit_fragment_image(repeat, (int(item_w - 20), int(panel_h - 65)))
    if mol_image is None:
        return False
    x = 32.0
    for index in range(visible):
        right = x + item_w
        draw.rounded_rectangle((x, top, right, top + panel_h), radius=10, fill=(255, 250, 229), outline=(0, 121, 107), width=2)
        draw.text((x + 10, top + 8), f"重复单元 {index + 1}", fill=(0, 105, 92), font=label_font)
        mol_x = int(x + (item_w - mol_image.width) / 2)
        mol_y = int(top + 36 + (panel_h - 60 - mol_image.height) / 2)
        image.paste(mol_image, (mol_x, mol_y))
        _draw_connector_caption(draw, repeat, x + 10, top + panel_h - 24, item_w - 20, note_font)
        x = right
        if index < visible - 1 or omitted:
            _draw_arrow(draw, x + 3, top + panel_h / 2, x + gap - 3, top + panel_h / 2)
            x += gap
    if omitted:
        right = x + item_w
        draw.rounded_rectangle((x, top, right, top + panel_h), radius=10, fill=(248, 250, 252), outline=(150, 160, 170), width=2)
        ellipsis_font = _load_schematic_font(25, bold=True)
        label = f"... x {omitted} ..."
        label_w, label_h = _text_size(draw, label, ellipsis_font)
        draw.text((x + (item_w - label_w) / 2, top + (panel_h - label_h) / 2), label, fill=(80, 95, 110), font=ellipsis_font)
    draw.text((32, height - 34), "BigSMILES 连接端保留为语义标记，采样图仅用于拓扑检查。", fill=(90, 90, 90), font=note_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return True

def _render_bigsmiles(raw: str, output_dir: Path, options: RenderOptions) -> RenderResult:
    parsed = _parse_bigsmiles(raw.strip())
    if not parsed.valid:
        return _invalid_result(raw, "bigsmiles", parsed.error, parsed.renderer)
    warnings = [
        "BigSMILES 主图表示重复单元/连接模式视图，不代表唯一完整聚合物分子",
        "二维位置用于结构检查和沟通，不代表真实聚合物构象",
    ]
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
