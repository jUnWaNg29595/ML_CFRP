# -*- coding: utf-8 -*-
"""通用外部模型特征补齐（Generic External-Model Feature Augmentation）

用途：
    在工作区之外训练好任意性能模型（Tg、交联密度、模量、强度……），
    用它对**工作区当前数据**做预测，并把预测值作为**新特征列**写回工作区，
    供后续训练/筛选使用。不针对任何特定目标列，任意模型通用。

与 core/data_imputer.py 的区别：
    data_imputer  —— 填充**已存在列**的缺失值（列必须已存在）
    本模块        —— **新增列**，并解决"模型特征名 ↔ 工作区列名"的对接问题

核心难点是**特征对接**：外部模型是按它自己那套列名训练的，工作区列名未必一致。
本模块用一条通用解析链解决（优先级从高到低）：

    1. 用户手工指定（manual_overrides）—— 最高优先级，UI 里可下拉选
    2. 精确同名
    3. 忽略大小写/空格同名
    4. 归一化同名（去前后缀、统一分隔符、单复数）
    5. 别名表（可外部 JSON 扩展，不写死业务知识）
    6. 模式推导（通用规则，见 _PATTERN_DERIVERS）
    7. 模糊匹配（difflib，相似度 ≥ 阈值，标记为"需确认"）
    8. 无法解析 → 报告，跳过该模型

设计原则：不写死任何业务字段。别名表和推导规则都可通过参数注入。
"""

from __future__ import annotations

import difflib
import json
import os
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .model_io import loads_artifact
except ImportError:  # pragma: no cover
    from model_io import loads_artifact


# ---------------------------------------------------------------------------
# 缺失值判定
# ---------------------------------------------------------------------------
_MISSING_TOKENS = {"", "nan", "none", "null", "na", "n/a", "-", "--", "unknown", "未测", "无", "?"}


def is_missing(series: pd.Series) -> pd.Series:
    """缺失判定：NaN / None / 空串 / 空白 / 常见占位符。"""
    if series.dtype.kind in "fiub":
        return series.isna()
    text = series.astype("string")
    stripped = text.str.strip().str.lower()
    return (
        series.isna()
        | stripped.isna()
        | stripped.isin(_MISSING_TOKENS)
    )


# ---------------------------------------------------------------------------
# 名称归一化（通用，不含业务知识）
# ---------------------------------------------------------------------------
_NORM_DROP_TOKENS = ("structure", "smiles", "bigsmiles", "value", "col", "column", "field")

_PLURAL_MAP = (
    ("_counts", "_count"),
    ("_groups", "_group"),
    ("_atoms", "_atom"),
    ("_bonds", "_bond"),
    ("_sites", "_site"),
    ("_rings", "_ring"),
    ("_ratios", "_ratio"),
    ("_values", "_value"),
    ("_names", "_name"),
    ("_types", "_type"),
)


def normalize_name(name: Any) -> str:
    """把列名归一化为可比较的骨架：小写、去非字母数字、去常见后缀词、单数化。"""
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    parts = [p for p in text.split("_") if p and p not in _NORM_DROP_TOKENS]
    text = "_".join(parts)
    for plural, singular in _PLURAL_MAP:
        if text.endswith(plural):
            text = text[: -len(plural)] + singular
            break
    return text


def clean_smiles(value: Any) -> Optional[str]:
    """去除 BigSMILES 包装与连接点标记，返回 RDKit 可解析的 SMILES。"""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    match = re.match(r"^\[\]\{(.*)\}\[\]$", text, flags=re.S)
    if match:
        text = match.group(1)
    text = re.sub(r"\[<[^\]]*\]|\[>[^\]]*\]", "", text)
    text = re.sub(r"\[\$\d*\]", "", text)
    text = re.sub(r"\{\[\]|\[\]\}", "", text)
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\[\]", "", text)
    return text.strip() or None


# ---------------------------------------------------------------------------
# 通用模式推导器
#   每个推导器：给定 df 和目标特征名，返回 Series 或 None（无法推导）
# ---------------------------------------------------------------------------
def _derive_component_count(df: pd.DataFrame, feature: str) -> Optional[pd.Series]:
    """通用规则：`{prefix}_component_count` ← 计数 `{prefix}_1_structure` / `_2_` / `_3_` …

    不限于已知前缀，任意前缀都适用（如 filler_component_count ← filler_1_structure）。
    """
    match = re.match(r"^(?P<prefix>.+?)_component_count$", feature)
    if not match:
        return None
    prefix = match.group("prefix")
    pattern = re.compile(rf"^{re.escape(prefix)}_\d+_?(structure|smiles|bigsmiles)$", re.I)
    sources = [c for c in df.columns if pattern.match(str(c))]
    if not sources:
        return None
    count = pd.Series(0, index=df.index, dtype="int64")
    for col in sources:
        count = count + (~is_missing(df[col])).astype("int64")
    return count


def _derive_structure_alias(df: pd.DataFrame, feature: str) -> Optional[pd.Series]:
    """通用规则：`{x}_structure` ← 同前缀的 `_smiles` / `_bigsmiles` 列（反之亦然）。"""
    match = re.match(r"^(?P<prefix>.+?)_(structure|smiles|bigsmiles)$", feature, re.I)
    if not match:
        return None
    prefix = match.group("prefix")
    for suffix in ("structure", "smiles", "bigsmiles"):
        for col in df.columns:
            if re.match(rf"^{re.escape(prefix)}_{suffix}$", str(col), re.I):
                return df[col]
    return None


def _derive_has_flag(df: pd.DataFrame, feature: str) -> Optional[pd.Series]:
    """通用规则：`{x}_present` / `has_{x}` ← `{x}_component_count`>0 或 `{x}_1_structure` 非空。"""
    match = re.match(r"^(?P<prefix>.+?)_present$", feature, re.I)
    if not match:
        match = re.match(r"^has_(?P<prefix>.+)$", feature, re.I)
    if not match:
        return None
    prefix = match.group("prefix")
    for col in df.columns:
        if re.match(rf"^{re.escape(prefix)}_component_count$", str(col), re.I):
            return pd.to_numeric(df[col], errors="coerce").fillna(0).gt(0)
    for col in df.columns:
        if re.match(rf"^{re.escape(prefix)}_\d+_?(structure|smiles)$", str(col), re.I):
            return ~is_missing(df[col])
    return None


def _derive_total_sum(df: pd.DataFrame, feature: str) -> Optional[pd.Series]:
    """通用规则：`{x}_total` ← 同前缀 `{x}_1_...`、`{x}_2_...` 数值列求和。"""
    match = re.match(r"^(?P<prefix>.+?)_total$", feature)
    if not match:
        return None
    prefix = match.group("prefix")
    pattern = re.compile(rf"^{re.escape(prefix)}_\d+_(?!structure|smiles|bigsmiles)(.+)$", re.I)
    sources = [c for c in df.columns if pattern.match(str(c))]
    if not sources:
        return None
    total = pd.Series(0.0, index=df.index)
    any_found = False
    for col in sources:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            total = total + values.fillna(0.0)
            any_found = True
    return total if any_found else None


#: 通用推导器注册表（顺序即优先级）。可通过 add_deriver() 扩展。
_PATTERN_DERIVERS: List[Tuple[str, Callable[[pd.DataFrame, str], Optional[pd.Series]]]] = [
    ("由组分结构列计数推导", _derive_component_count),
    ("由同前缀结构列映射", _derive_structure_alias),
    ("由组分存在性推导", _derive_has_flag),
    ("由同前缀数值列求和", _derive_total_sum),
]


def add_deriver(label: str, func: Callable[[pd.DataFrame, str], Optional[pd.Series]], *, front: bool = False) -> None:
    """注册自定义推导器（便于项目扩展，无需改动本模块）。"""
    entry = (label, func)
    if front:
        _PATTERN_DERIVERS.insert(0, entry)
    else:
        _PATTERN_DERIVERS.append(entry)


# ---------------------------------------------------------------------------
# 别名表：可外部 JSON 覆盖，默认留空（不写死业务字段）
# ---------------------------------------------------------------------------
DEFAULT_ALIAS_TABLE: Dict[str, List[str]] = {}


def load_alias_table(path: str | os.PathLike[str]) -> Dict[str, List[str]]:
    """从 JSON 载入别名表：{"外部特征名": ["工作区候选列名", ...]}"""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("别名表必须是 JSON 对象：{特征名: [候选列名, ...]}")
    table: Dict[str, List[str]] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            table[str(key)] = [value]
        elif isinstance(value, (list, tuple)):
            table[str(key)] = [str(v) for v in value]
    return table


def save_alias_table(table: Dict[str, Sequence[str]], path: str | os.PathLike[str]) -> None:
    payload = {str(k): list(v) if not isinstance(v, str) else [v] for k, v in table.items()}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 解析结果
# ---------------------------------------------------------------------------
RESOLVE_STRATEGIES = ("manual", "exact", "case_insensitive", "normalized", "alias", "pattern", "fuzzy")


class FeatureResolution(dict):
    """解析结果容器（dict 子类，便于直接序列化/展示）。"""

    @property
    def resolved(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for strategy in RESOLVE_STRATEGIES:
            out.update(self.get(strategy) or {})
        return out

    @property
    def unresolved(self) -> List[str]:
        return list(self.get("unresolved") or [])

    @property
    def needs_review(self) -> Dict[str, str]:
        """模糊匹配的结果——能跑但建议人工确认。"""
        return dict(self.get("fuzzy") or {})

    @property
    def derived(self) -> Dict[str, str]:
        return {k: v for k, v in (self.get("pattern") or {}).items()}


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------
class ExternalFeatureAugmenter:
    """通用外部模型特征补齐器：任意模型 → 预测 → 新特征列。"""

    def __init__(
        self,
        artifacts: Sequence[bytes],
        *,
        model_names: Optional[Sequence[str]] = None,
        alias_table: Optional[Dict[str, Sequence[str]]] = None,
        fuzzy_threshold: float = 0.86,
    ):
        """
        参数:
            artifacts:       模型文件字节序列（joblib 序列化的 artifact）
            model_names:     可选显示名，与 artifacts 一一对应
            alias_table:     可选别名表 {外部特征名: [候选列名, ...]}
            fuzzy_threshold: 模糊匹配阈值（0-1），越高越保守
        """
        if not artifacts:
            raise ValueError("至少需要提供一个模型文件")
        self.alias_table: Dict[str, List[str]] = {
            str(k): list(v) if not isinstance(v, str) else [v]
            for k, v in (alias_table or DEFAULT_ALIAS_TABLE).items()
        }
        self.fuzzy_threshold = float(fuzzy_threshold)
        self.entries: List[Dict[str, Any]] = []

        for idx, blob in enumerate(artifacts):
            artifact = loads_artifact(blob)
            predictor = artifact.get("pipeline")
            if predictor is None:
                predictor = artifact.get("model")
            if predictor is None:
                raise ValueError(f"第 {idx + 1} 个模型文件无效：缺少 pipeline 或 model")

            target = str(artifact.get("target_col") or "").strip()
            if not target:
                raise ValueError(f"第 {idx + 1} 个模型文件缺少 target_col，无法确定输出列名")

            name = None
            if model_names and idx < len(model_names):
                name = model_names[idx]
            name = name or artifact.get("model_name") or target

            feature_cols = [str(c) for c in (artifact.get("feature_cols") or [])]
            self.entries.append(
                {
                    "index": idx,
                    "name": str(name),
                    "artifact": artifact,
                    "predictor": predictor,
                    "target_col": target,
                    "feature_cols": feature_cols,
                    "metrics": dict(artifact.get("metrics") or {}),
                    "extra": dict(artifact.get("extra") or {}),
                }
            )

    # -- 内省 ---------------------------------------------------------------
    def get_info(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": e["name"],
                "target_col": e["target_col"],
                "n_features": len(e["feature_cols"]),
                "feature_cols": list(e["feature_cols"]),
                "metrics": dict(e["metrics"]),
                "output_col": f"{e['target_col']}_pred",
                "model_type": type(e["predictor"]).__name__,
            }
            for e in self.entries
        ]

    def required_features(self) -> List[str]:
        """所有模型需要的特征名并集（保持首次出现顺序）。"""
        seen: List[str] = []
        for entry in self.entries:
            for col in entry["feature_cols"]:
                if col not in seen:
                    seen.append(col)
        return seen

    # -- 特征解析 -----------------------------------------------------------
    def resolve_features(
        self,
        df: pd.DataFrame,
        *,
        manual_overrides: Optional[Dict[str, str]] = None,
        allow_fuzzy: bool = True,
        features: Optional[Sequence[str]] = None,
    ) -> FeatureResolution:
        """把模型特征名映射到工作区实际列名。

        参数:
            manual_overrides: {外部特征名: 工作区列名}，优先级最高
            allow_fuzzy:      是否启用模糊匹配（关闭则未匹配项直接进 unresolved）
            features:         只解析这些特征（默认解析全部模型的并集）。
                              按模型分别解析可避免"一个模型缺特征连坐其他模型"。
        """
        manual_overrides = {str(k): str(v) for k, v in (manual_overrides or {}).items()}
        columns = [str(c) for c in df.columns]
        case_map: Dict[str, str] = {}
        norm_map: Dict[str, str] = {}
        for col in columns:
            case_map.setdefault(col.strip().lower(), col)
            norm_map.setdefault(normalize_name(col), col)

        result = FeatureResolution(
            manual={}, exact={}, case_insensitive={}, normalized={}, alias={},
            pattern={}, fuzzy={}, unresolved=[], pattern_notes={},
        )

        wanted = list(features) if features is not None else self.required_features()
        for feature in wanted:
            # 1) 手工指定
            override = manual_overrides.get(feature)
            if override and override in columns:
                result["manual"][feature] = override
                continue
            # 2) 精确
            if feature in columns:
                result["exact"][feature] = feature
                continue
            # 3) 忽略大小写/空格
            hit = case_map.get(feature.strip().lower())
            if hit:
                result["case_insensitive"][feature] = hit
                continue
            # 4) 归一化
            hit = norm_map.get(normalize_name(feature))
            if hit:
                result["normalized"][feature] = hit
                continue
            # 5) 别名表
            alias_hit = None
            for candidate in self.alias_table.get(feature, []):
                if candidate in columns:
                    alias_hit = candidate
                    break
                alias_hit = case_map.get(str(candidate).strip().lower()) or norm_map.get(normalize_name(candidate))
                if alias_hit:
                    break
            if alias_hit:
                result["alias"][feature] = alias_hit
                continue
            # 6) 通用模式推导
            derived = None
            for label, func in _PATTERN_DERIVERS:
                try:
                    series = func(df, feature)
                except Exception:
                    series = None
                if series is not None:
                    derived = (label, series)
                    break
            if derived is not None:
                result["pattern"][feature] = derived[0]
                result["pattern_notes"][feature] = derived[0]
                continue
            # 7) 模糊匹配
            if allow_fuzzy:
                pool = list(case_map.keys()) + list(norm_map.keys())
                matches = difflib.get_close_matches(normalize_name(feature), pool, n=1, cutoff=self.fuzzy_threshold)
                if matches:
                    key = matches[0]
                    result["fuzzy"][feature] = case_map.get(key) or norm_map.get(key)
                    continue
            result["unresolved"].append(feature)

        return result

    # -- 特征矩阵构造 -------------------------------------------------------
    def build_feature_frame(
        self,
        df: pd.DataFrame,
        resolution: FeatureResolution,
        entry: Optional[Dict[str, Any]] = None,
        *,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """按模型要求构造输入矩阵（列名与顺序严格对齐模型训练时的 feature_cols）。

        非数值列（如原始 SMILES）无法直接送入模型：外部模型必须自带特征化
        （例如把 RDKit 描述符计算包进 Pipeline）。此处只做数值强制转换，
        转换失败则保留 NaN 并在 augment 阶段报告，避免 sklearn 抛出难懂的异常。
        """
        wanted = list(feature_cols if feature_cols is not None else self.required_features())
        mapping = resolution.resolved
        pattern_keys = resolution.get("pattern") or {}
        frame = pd.DataFrame(index=df.index)
        for feature in wanted:
            if feature in pattern_keys:
                series = None
                for _label, func in _PATTERN_DERIVERS:
                    series = func(df, feature)
                    if series is not None:
                        break
                frame[feature] = series if series is not None else np.nan
            elif feature in mapping:
                frame[feature] = df[mapping[feature]]
            else:
                frame[feature] = np.nan

        # 数值强制转换：object/string 列转数值，不可转的置 NaN（避免模型直接报错）
        for col in frame.columns:
            if frame[col].dtype.kind in "fiub":
                continue
            converted = pd.to_numeric(frame[col], errors="coerce")
            # 保留可解释的类别列：若原本是字符串且转换后全空，则编码为整数码
            if converted.notna().sum() == 0 and frame[col].notna().any():
                frame[col] = pd.Categorical(frame[col]).codes.astype(float)
            else:
                frame[col] = converted
        return frame

    def diagnose_entry(self, df: pd.DataFrame, entry: Dict[str, Any]) -> Dict[str, Any]:
        """诊断单个模型能否在当前数据上运行（不预测，供 UI 预览）。"""
        feature_cols = entry["feature_cols"]
        resolution = self.resolve_features(df, features=feature_cols)
        mapping = resolution.resolved
        pattern_keys = resolution.get("pattern") or {}
        rows: List[Dict[str, Any]] = []
        for feature in feature_cols:
            if feature in pattern_keys:
                rows.append({"feature": feature, "source": f"<推导: {pattern_keys[feature]}>", "strategy": "pattern"})
            elif feature in mapping:
                strategy = next(
                    (s for s in RESOLVE_STRATEGIES if feature in (resolution.get(s) or {})), "unknown"
                )
                src = mapping[feature]
                fill = float((~is_missing(df[src])).mean()) if src in df.columns else 0.0
                rows.append({"feature": feature, "source": src, "strategy": strategy, "fill_rate": fill})
            else:
                rows.append({"feature": feature, "source": None, "strategy": "unresolved"})
        n_ok = sum(1 for r in rows if r["strategy"] != "unresolved")
        return {
            "name": entry["name"],
            "target_col": entry["target_col"],
            "output_col": f"{entry['target_col']}_pred",
            "features": rows,
            "n_required": len(feature_cols),
            "n_resolved": n_ok,
            "feature_coverage": n_ok / max(1, len(feature_cols)),
            "unresolved": [r["feature"] for r in rows if r["strategy"] == "unresolved"],
            "needs_review": dict(resolution.needs_review),
        }

    # -- 主流程 -------------------------------------------------------------
    def augment(
        self,
        df: pd.DataFrame,
        *,
        manual_overrides: Optional[Dict[str, str]] = None,
        output_mode: str = "new_column",
        suffix: str = "_pred",
        add_source_flag: bool = True,
        allow_fuzzy: bool = True,
        allow_partial: bool = True,
        min_feature_coverage: float = 0.5,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """预测并写回。

        参数:
            df:                工作区数据
            manual_overrides:  {外部特征名: 工作区列名}
            output_mode:
                "new_column"   —— 总是写入 `{target}{suffix}` 新列（默认）
                "fill_missing" —— 目标列缺失的行写入目标列本身，其余保留
                "overwrite"    —— 用预测值覆盖目标列全部行
            suffix:            新列后缀
            add_source_flag:   是否附带 `{col}_source` 标记 observed/predicted
            allow_fuzzy:       是否允许模糊匹配特征名
            allow_partial:     True（默认）—— 部分特征缺失时仍预测，缺失列传 NaN，
                               交由模型自带的 imputer 处理（sklearn Pipeline 常见）
            min_feature_coverage: allow_partial 时的最低特征覆盖率（低于此值仍跳过）

        返回:
            (augmented_df, reports)
        """
        if output_mode not in ("new_column", "fill_missing", "overwrite"):
            raise ValueError("output_mode 必须是 new_column / fill_missing / overwrite 之一")

        result = df.copy()
        reports: List[Dict[str, Any]] = []

        for entry in self.entries:
            target = entry["target_col"]
            feature_cols = entry["feature_cols"]
            out_col = target if output_mode != "new_column" else f"{target}{suffix}"

            # 按模型分别解析，避免"一个模型缺特征连坐其他模型"
            resolution = self.resolve_features(
                result, manual_overrides=manual_overrides, allow_fuzzy=allow_fuzzy, features=feature_cols,
            )
            n_resolved = sum(
                1 for f in feature_cols
                if f in resolution.resolved or f in (resolution.get("pattern") or {})
            )
            coverage_ratio = n_resolved / max(1, len(feature_cols))
            report: Dict[str, Any] = {
                "name": entry["name"],
                "target_col": target,
                "output_col": out_col,
                "output_mode": output_mode,
                "status": "ok",
                "n_features_required": len(feature_cols),
                "n_features_resolved": n_resolved,
                "feature_coverage": float(coverage_ratio),
                "unresolved": list(resolution.unresolved),
                "needs_review": dict(resolution.needs_review),
                "n_predicted": 0,
                "n_observed": 0,
                "n_total": int(len(result)),
                "coverage": 0.0,
                "pred_min": None,
                "pred_max": None,
                "pred_mean": None,
                "note": "",
            }

            if resolution.unresolved and not allow_partial:
                report["status"] = "skipped"
                report["note"] = f"缺少 {len(resolution.unresolved)} 个必需特征，已跳过（可在界面手工映射）"
                reports.append(report)
                continue
            if coverage_ratio < min_feature_coverage:
                report["status"] = "skipped"
                report["note"] = (
                    f"特征覆盖仅 {coverage_ratio*100:.0f}%（低于阈值 {min_feature_coverage*100:.0f}%），已跳过"
                )
                reports.append(report)
                continue

            try:
                features = self.build_feature_frame(result, resolution, entry, feature_cols=feature_cols)
            except Exception as exc:
                report["status"] = "error"
                report["note"] = f"构造特征失败: {exc}"
                reports.append(report)
                continue

            # 缺失特征填 NaN：模型自带的 imputer 会处理；若模型无 imputer 会在此抛错
            features = features.reindex(columns=feature_cols)
            if resolution.unresolved:
                report["note"] = (
                    f"{len(resolution.unresolved)} 个特征缺失已置 NaN，交由模型内置填充处理"
                )

            if output_mode == "new_column":
                observed_mask = pd.Series(False, index=result.index)
                predict_mask = pd.Series(True, index=result.index)
            else:
                if target in result.columns:
                    missing_mask = is_missing(result[target])
                else:
                    # 目标列不存在：视为全缺失，预测后创建该列
                    result[target] = np.nan
                    missing_mask = pd.Series(True, index=result.index)
                observed_mask = ~missing_mask
                # overwrite: 覆盖全部行；fill_missing: 只补缺失行
                predict_mask = pd.Series(True, index=result.index) if output_mode == "overwrite" else missing_mask

            report["n_observed"] = int(observed_mask.sum())

            if predict_mask.sum() == 0:
                if output_mode == "new_column":
                    result[out_col] = (
                        pd.to_numeric(result[target], errors="coerce") if target in result.columns else np.nan
                    )
                elif target not in result.columns:
                    # 目标列不存在：创建空列，避免后续 KeyError
                    result[target] = np.nan
                report["status"] = "noop"
                report["note"] = "没有需要预测的行（目标列已全部有值）"
                report["coverage"] = float(is_missing(result[out_col]).eq(False).mean()) if out_col in result else 0.0
                reports.append(report)
                continue

            try:
                preds = np.asarray(entry["predictor"].predict(features.loc[predict_mask]), dtype=float).reshape(-1)
            except Exception as exc:
                report["status"] = "error"
                hint = ""
                msg = str(exc)
                if "could not convert string to float" in msg or "non-numeric" in msg.lower():
                    hint = (
                        "\n提示：该模型需要数值特征，但输入中存在无法转换的文本列。"
                        "外部模型需自带特征化（把 RDKit 描述符计算包进 Pipeline），"
                        "否则结构列（SMILES）无法直接作为模型输入。"
                    )
                report["note"] = f"预测失败: {msg}{hint}"
                reports.append(report)
                continue

            if output_mode == "new_column":
                out = pd.Series(np.nan, index=result.index, dtype=float)
                out.loc[predict_mask] = preds
            elif output_mode == "overwrite":
                out = pd.Series(np.nan, index=result.index, dtype=float)
                if target in result.columns:
                    out = pd.to_numeric(result[target], errors="coerce").astype(float)
                out.loc[predict_mask] = preds
            else:  # fill_missing
                out = pd.Series(np.nan, index=result.index, dtype=float)
                if target in result.columns:
                    out = pd.to_numeric(result[target], errors="coerce").astype(float)
                out.loc[predict_mask] = preds

            result[out_col] = out

            if add_source_flag:
                flag = pd.Series("observed", index=result.index, dtype=object)
                if output_mode == "new_column":
                    flag[:] = "predicted"
                elif output_mode == "overwrite":
                    flag.loc[predict_mask] = "predicted"
                else:
                    flag.loc[predict_mask] = "predicted"
                result[f"{out_col}_source"] = flag

            report["n_predicted"] = int(predict_mask.sum())
            report["coverage"] = float(pd.Series(out).notna().mean())
            if len(preds):
                report["pred_min"] = float(np.nanmin(preds))
                report["pred_max"] = float(np.nanmax(preds))
                report["pred_mean"] = float(np.nanmean(preds))
            reports.append(report)

        return result, reports


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------
def augment_with_models(
    df: pd.DataFrame,
    model_blobs: Sequence[bytes],
    *,
    model_names: Optional[Sequence[str]] = None,
    alias_table: Optional[Dict[str, Sequence[str]]] = None,
    manual_overrides: Optional[Dict[str, str]] = None,
    output_mode: str = "new_column",
    suffix: str = "_pred",
    add_source_flag: bool = True,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """一次性用多个外部模型补齐特征列（通用）。"""
    augmenter = ExternalFeatureAugmenter(model_blobs, model_names=model_names, alias_table=alias_table)
    return augmenter.augment(
        df,
        manual_overrides=manual_overrides,
        output_mode=output_mode,
        suffix=suffix,
        add_source_flag=add_source_flag,
    )


def augment_with_model_paths(df: pd.DataFrame, model_paths: Sequence[str], **kwargs: Any) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """按文件路径读取模型并补齐。"""
    blobs: List[bytes] = []
    for path in model_paths:
        with open(path, "rb") as handle:
            blobs.append(handle.read())
    return augment_with_models(df, blobs, **kwargs)


def describe_models(model_blobs: Sequence[bytes]) -> List[Dict[str, Any]]:
    """只读取模型元信息（不预测），用于 UI 预览。"""
    return ExternalFeatureAugmenter(model_blobs).get_info()
