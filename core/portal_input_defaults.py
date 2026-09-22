"""门户输入默认值读取（离线统计产物 ``portal_input_defaults.json`` 的访问层）。

设计依据：docs/superpowers/specs/2026-09-22-portal-formulation-first-input-design.md §4.2

职责边界
--------
本模块**只读取**离线统计产物，不做任何统计、不推断、不补默认值。

硬约束（spec 明令，代码内强制）
------------------------------
1. **只对 ``manual_input`` 分区字段返回默认值**；
   ``derived_workflow`` / ``molecular_workflow`` 字段一律返回 ``None``。
   原因：这些字段必须由 workflow 从 SMILES / cure_schedule 真实计算，
   提供默认值等同于伪造数据，会造成静默的预测错误。
2. 任何异常（文件缺失、JSON 损坏、结构不符）都降级为"无默认值"，
   **不得抛出** —— 门户不能因缺文件而崩溃。
3. 默认值一律携带 ``share`` / ``support`` / ``source_table``，供 UI 展示审计依据。

产物由 ``scripts/build_portal_input_defaults.py`` 生成并随代码版本化提交。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "DEFAULTS_FILENAME",
    "default_for_feature",
    "defaults_for_target",
    "load_portal_input_defaults",
    "recipe_defaults",
]

#: 产物文件名（位于 ``prediction_portal/`` 下）
DEFAULTS_FILENAME = "portal_input_defaults.json"

#: 唯一允许获得默认值的分区。其余分区一律返回 None。
MANUAL_PARTITION = "manual_input"

#: 进程内缓存：{绝对路径: (mtime_ns, payload)}，避免每次渲染都读盘
_CACHE: dict[str, tuple[int, dict[str, Any]]] = {}


def _defaults_path(root: str | Path | None = None) -> Path:
    """解析产物路径。``root`` 可以是项目根，也可以直接是 prediction_portal 目录。"""
    if root is None:
        base = Path(__file__).resolve().parents[1]
    else:
        base = Path(root).expanduser()
    if base.name == "prediction_portal":
        return base / DEFAULTS_FILENAME
    candidate = base / "prediction_portal" / DEFAULTS_FILENAME
    if candidate.is_file() or not (base / DEFAULTS_FILENAME).is_file():
        return candidate
    return base / DEFAULTS_FILENAME


def load_portal_input_defaults(root: str | Path | None = None) -> dict[str, Any]:
    """加载产物；缺失或损坏时返回空字典（不抛异常）。"""
    path = _defaults_path(root)
    try:
        stat = path.stat()
    except OSError:
        return {}
    key = str(path)
    cached = _CACHE.get(key)
    if cached and cached[0] == stat.st_mtime_ns:
        return cached[1]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    _CACHE[key] = (stat.st_mtime_ns, payload)
    return payload


def _fields_block(payload: Mapping[str, Any], target_col: str) -> dict[str, Any]:
    defaults = payload.get("defaults")
    if not isinstance(defaults, Mapping):
        return {}
    block = defaults.get(str(target_col))
    if not isinstance(block, Mapping):
        return {}
    fields = block.get("fields")
    return dict(fields) if isinstance(fields, Mapping) else {}


def defaults_for_target(
    target_col: str, root: str | Path | None = None
) -> dict[str, Any]:
    """返回某目标列的测试条件默认值集合 ``{feature_name: record}``。

    仅返回**测试条件类**字段（由 ``defaults`` 分区提供）；
    配方聚合类默认值请用 :func:`recipe_defaults`。
    """
    payload = load_portal_input_defaults(root)
    return _fields_block(payload, target_col)


def recipe_defaults(root: str | Path | None = None) -> dict[str, Any]:
    """返回配方聚合类默认值集合 ``{feature_name: record}``。"""
    payload = load_portal_input_defaults(root)
    block = payload.get("recipe_defaults")
    return dict(block) if isinstance(block, Mapping) else {}


def default_for_feature(
    feature: str,
    *,
    partition: str,
    target_col: str = "",
    root: str | Path | None = None,
) -> dict[str, Any] | None:
    """查某字段的默认值记录；**仅当分区为 ``manual_input`` 时**才可能返回。

    参数
    ----
    feature : 字段名（contract 中的特征列名）
    partition : 该字段在契约/注册表中的分区（``manual_input`` /
        ``derived_workflow`` / ``molecular_workflow``）
    target_col : 目标列名，用于定位测试条件默认值；配方类字段可留空

    返回
    ----
    命中时返回带 ``value`` / ``share`` / ``support`` / ``source_table`` 的字典；
    否则返回 ``None``。

    注意：非 ``manual_input`` 分区**直接返回 None**，不查表 —— 这是 spec 的
    硬约束，避免把"系统计算特征"用统计默认值伪造出来。
    """
    if str(partition or "").strip() != MANUAL_PARTITION:
        return None
    name = str(feature or "").strip()
    if not name:
        return None

    payload = load_portal_input_defaults(root)

    if target_col:
        record = _fields_block(payload, target_col).get(name)
        if isinstance(record, Mapping):
            return dict(record)

    # 配方聚合类字段（无 target_col 前缀）
    record = recipe_defaults(root).get(name)
    if isinstance(record, Mapping):
        return dict(record)
    return None
