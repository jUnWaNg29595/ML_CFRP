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

**级联模型支持**（Cascaded Models）：
    有些模型的特征列本身就是**别的模型的预测目标**。例如：

        XGBoost_artifact(2).joblib      target=tg_c               需要 tensile_modulus_gpa
        拉伸模量-XGBoost-0.965.joblib   target=tensile_modulus_gpa 需要 tg_c

    这两个模型**互相引用**，单独导入任一个都跑不起来。本模块自动：

        1. 发现依赖：模型 B 的输入特征名 == 模型 A 的 target_col
        2. 拓扑分层：用 Tarjan SCC + 缩点拓扑排序，算出求解层
        3. 逐层求解：先算上游、把预测值写回，作为下游模型的输入特征
        4. 环处理：多节点 SCC 无法拓扑排序，按“依赖的**环外可获得性**”
           择优打破——优先用工作区真值/手工映射，其次比可解析特征比例

    优先级铁律：**工作区已有的真实值一定胜过模型预测值**。
    级联解析排在 exact/case_insensitive/normalized/alias 之后、pattern/fuzzy 之前。

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
# ---------------------------------------------------------------------------
# 模型级联依赖（级联模型 / Cascaded Models）
# ---------------------------------------------------------------------------
# 背景：有些模型的特征列本身就是**别的模型的预测目标**。
#   例：
#     XGBoost_artifact(2).joblib      target=tg_c              feature_cols 含 tensile_modulus_gpa
#     拉伸模量-XGBoost-0.965.joblib   target=tensile_modulus_gpa  feature_cols 含 tg_c
#   单独导入任何一个都跑不起来（缺的特征谁也补不出来），必须**成链求解**。
#   而且这两个是**互相引用**，构成环，必须打破环（用可测/可查的先算，或
#   直接用总表/工作区已有值，或按用户指定的顺序强制解）。
#
# 本模块提供的机制：
#   1. 自动发现依赖：模型 B 的输入特征名 == 模型 A 的 target_col
#   2. 拓扑分层：能解的先解，解完把预测值写回，作为下游模型的输入
#   3. 环处理：SCC（强连通分量）内按“外部可获得性”择优打破，
#      并把剩余列交给特征解析链（总表/现场计算/手工映射）

# 判定“某特征是否其实是另一个模型的预测目标”时允许的别名后缀。
# 模型 A 的 target 是 tg_c，模型 B 的输入列可能写成 tg_c_pred / predicted_tg_c。
_DEP_TARGET_SUFFIXES = ("_pred", "_predicted", "_prediction", "_value", "_est", "_estimated")
_DEP_TARGET_PREFIXES = ("pred_", "predicted_", "prediction_", "est_", "estimated_")


def _dep_key(name: Any) -> str:
    """把特征名/目标名归一化为可比较的依赖键（去预测后缀/前缀 + normalize）。"""
    text = str(name).strip()
    low = text.lower()
    for pre in _DEP_TARGET_PREFIXES:
        if low.startswith(pre):
            text = text[len(pre):]
            low = text.lower()
            break
    for suf in _DEP_TARGET_SUFFIXES:
        if low.endswith(suf):
            text = text[: -len(suf)]
            break
    return normalize_name(text)


# 解析策略优先级（从高到低）。注意 cascade（上游模型预测值）排在
# 真实列匹配（manual/exact/case/normalized/alias）之后、推导/模糊之前：
# **工作区已有的真实值一定胜过模型预测值**。
# 方法名 → 产物列名特征子串（用于前缀命中时判定“这列真是这步算的”）
# 目的：防止短前缀陷阱。例：指纹步的 prefix='resin_' 会命中
# `resin_3_f_stoich`（化学计量列，不是指纹），从而把该步误保留。
_METHOD_OUTPUT_MARKERS: List[Tuple[str, Tuple[str, ...]]] = [
    ("指纹", ("MACCS", "Morgan", "ECFP", "FCFP", "FP_", "Fingerprint")),
    ("xTB", ("_xtb_", "xtb_")),
    ("快速力场", ("mmff", "uff", "forcefield", "_ff_")),
    ("环氧树脂反应", ("crosslink", "reaction", "stoich", "mechanism")),
    ("3D", ("_3d_", "npr", "spherocity", "pbf", "coulomb")),
    ("Mordred", ("mordred", "AATS", "BCUT", "EState", "PEOE", "VSA", "TopoPSA")),
    ("FGD", ("fgd", "functional_group")),
]


def _is_method_output(method: str, candidates: Sequence[str]) -> bool:
    """判断候选列名里是否真存在“该方法算出来的”列。

    用于 `_prune_workflow_to_needed_steps` 的前缀命中校验：
    只知道 prefix 不够（短前缀会误命中），还得列名里有该方法的标志性片段。

    方法名无法识别时返回 True（保守，宁可多跑）。
    """
    m = str(method or "").lower()
    markers: Tuple[str, ...] = ()
    for key, pats in _METHOD_OUTPUT_MARKERS:
        if key.lower() in m:
            markers = pats
            break
    if not markers:
        return True
    lows = [str(c).lower() for c in candidates]
    return any(any(p.lower() in c for p in markers) for c in lows)


RESOLVE_STRATEGIES = ("manual", "exact", "case_insensitive", "normalized", "alias", "cascade", "pattern", "fuzzy")


class ModelDependencyGraph:
    """模型之间的特征依赖图：谁的特征要等谁的预测。

    节点 = 模型 index；边 A → B 表示「A 的某个输入特征是 B 的 target」，
    即 **A 依赖 B**，B 必须先算。
    """

    def __init__(self, entries: Sequence[Dict[str, Any]]):
        self.entries = list(entries)
        self.n = len(self.entries)
        # target 键 → 模型 index（多个模型同 target 时取第一个）
        self.target_index: Dict[str, int] = {}
        for i, e in enumerate(self.entries):
            tgt = e.get("target_col")
            if tgt:
                self.target_index.setdefault(_dep_key(tgt), i)
        # 每个模型的外部依赖：{模型i: {依赖键: 提供它的模型j}}
        self.deps: Dict[int, Dict[str, int]] = {}
        # 反向：{模型j: [依赖它的模型i, ...]}
        self.dependents: Dict[int, List[int]] = {i: [] for i in range(self.n)}
        for i, e in enumerate(self.entries):
            cols = list(e.get("input_feature_cols") or e.get("feature_cols") or [])
            own = _dep_key(e.get("target_col") or "")
            found: Dict[str, int] = {}
            for col in cols:
                key = _dep_key(col)
                if not key or key == own:
                    continue
                j = self.target_index.get(key)
                if j is not None and j != i:
                    found.setdefault(key, j)
            self.deps[i] = found
            for j in set(found.values()):
                self.dependents[j].append(i)

    # -- 查询 ---------------------------------------------------------------
    def has_any_dependency(self) -> bool:
        return any(self.deps.get(i) for i in range(self.n))

    def upstream_of(self, i: int) -> List[int]:
        """i 依赖的模型（需先算）。"""
        return sorted(set(self.deps.get(i, {}).values()))

    def is_self_contained(self, i: int) -> bool:
        return not self.deps.get(i)

    def describe(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i, e in enumerate(self.entries):
            ups = self.upstream_of(i)
            if not ups:
                continue
            out.append({
                "model": e.get("name"),
                "target_col": e.get("target_col"),
                "depends_on": [
                    {
                        "feature": col,
                        "provided_by": self.entries[j].get("name"),
                        "provider_target": self.entries[j].get("target_col"),
                    }
                    for col, j in sorted(self.deps[i].items())
                ],
            })
        return out

    # -- 分层 ---------------------------------------------------------------
    def _sccs(self) -> List[List[int]]:
        """Tarjan 求强连通分量（SCC）。多节点 SCC = 互相依赖的环。"""
        index_of: Dict[int, int] = {}
        low: Dict[int, int] = {}
        on_stack: Dict[int, bool] = {}
        stack: List[int] = []
        sccs: List[List[int]] = []
        counter = [0]

        def strongconnect(v: int) -> None:
            index_of[v] = low[v] = counter[0]
            counter[0] += 1
            stack.append(v)
            on_stack[v] = True
            for w in self.upstream_of(v):
                if w not in index_of:
                    strongconnect(w)
                    low[v] = min(low[v], low[w])
                elif on_stack.get(w):
                    low[v] = min(low[v], index_of[w])
            if low[v] == index_of[v]:
                comp: List[int] = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp.append(w)
                    if w == v:
                        break
                sccs.append(sorted(comp))

        for v in range(self.n):
            if v not in index_of:
                strongconnect(v)
        return sccs

    def scc_layers(self) -> List[List[List[int]]]:
        """把 SCC 缩点后拓扑排序，返回分层结果。

        返回 `[[[0,1], [2]], [[3]]]` 表示：
          第 0 层可并行求解 {0,1}（互引环）和 {2}（独立），
          第 1 层是 {3}（依赖前一层）。
        层内各 SCC 之间无依赖，可以任意顺序；SCC 内部若是环则需要打破。
        """
        sccs = self._sccs()
        comp_of: Dict[int, int] = {}
        for ci, comp in enumerate(sccs):
            for v in comp:
                comp_of[v] = ci

        # 缩点图的依赖：comp A 依赖 comp B 当且仅当存在 v∈A, w∈B 且 v 依赖 w
        comp_deps: Dict[int, set] = {ci: set() for ci in range(len(sccs))}
        for v in range(self.n):
            for w in self.upstream_of(v):
                if comp_of[v] != comp_of[w]:
                    comp_deps[comp_of[v]].add(comp_of[w])

        # Kahn 拓扑分层
        remaining = {ci: set(d) for ci, d in comp_deps.items()}
        out: List[List[List[int]]] = []
        done: set = set()
        while remaining:
            ready = sorted(ci for ci, d in remaining.items() if not (d - done))
            if not ready:
                ready = sorted(remaining)  # 兜底（已缩点，理论不可达）
            out.append([sccs[ci] for ci in ready])
            for ci in ready:
                done.add(ci)
                remaining.pop(ci)
        return out

    def layers(self) -> List[List[int]]:
        """扁平化的分层（层内模型互不依赖，可任意顺序）。"""
        return [[v for comp in layer for v in comp] for layer in self.scc_layers()]

    def cycles(self) -> List[List[int]]:
        """返回互相依赖的模型组（真实 SCC，每组 size ≥ 2）。"""
        return [comp for comp in self._sccs() if len(comp) > 1]


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
            # 关键：模型 pipeline 的真实输入列数可能大于 artifact.feature_cols。
            # 例：Pipeline(imputer -> feature_mask -> scaler -> model)，
            #     imputer 吃 2070 列（canonical），mask 后 1408 列进 model，
            #     而 artifact.feature_cols 只记录了 mask 后的 1408 列。
            # 这时必须按 imputer 的期望列数（2070）喂数据，否则报
            # "X has 1408 features, but SimpleImputer is expecting 2070"。
            input_feature_cols = self._resolve_input_feature_cols(
                artifact, predictor, feature_cols
            )
            # 宽度不匹配时（imputer 652 vs 我们只有 515 列），用 mask 算出散布索引：
            # 列名无关、位置才重要，缺位填 NaN 交给 imputer。
            _mask = self._pipeline_mask(predictor)
            _expander = self._resolve_input_expander(predictor, input_feature_cols, _mask)
            _expected = self._pipeline_expected_n_features(predictor)
            self.entries.append(
                {
                    "index": idx,
                    "name": str(name),
                    "artifact": artifact,
                    "predictor": predictor,
                    "target_col": target,
                    "feature_cols": feature_cols,
                    "input_feature_cols": input_feature_cols,
                    "input_expander": _expander,
                    "pipeline_n_features": _expected,
                    "metrics": dict(artifact.get("metrics") or {}),
                    "extra": dict(artifact.get("extra") or {}),
                }
            )

        # 构建模型间依赖图（级联模型支持）：谁的输入特征其实是别人的预测目标。
        # 必须在所有 entry 就位后构建，因为要按 target_col 互相对照。
        self.dependency_graph = ModelDependencyGraph(self.entries)
        for i, entry in enumerate(self.entries):
            entry["depends_on_models"] = self.dependency_graph.upstream_of(i)
            entry["cascade_inputs"] = dict(self.dependency_graph.deps.get(i) or {})

    @staticmethod
    def _pipeline_expected_n_features(predictor: Any) -> Optional[int]:
        """取 pipeline 首步声明的输入列数（sklearn 在 fit 时记录）。"""
        try:
            for _name, step in (getattr(predictor, "steps", None) or []):
                n = getattr(step, "n_features_in_", None)
                if n:
                    return int(n)
        except Exception:
            pass
        return None

    @staticmethod
    def _pipeline_mask(predictor: Any) -> Optional[np.ndarray]:
        """取 pipeline 里 FeatureMaskTransformer 的布尔掩码（按位置生效）。"""
        try:
            for _name, step in (getattr(predictor, "steps", None) or []):
                m = getattr(step, "feature_mask", None)
                if m is not None:
                    return np.asarray(m, dtype=bool).reshape(-1)
        except Exception:
            pass
        return None

    @staticmethod
    def _resolve_input_expander(
        predictor: Any,
        input_cols: List[str],
        mask: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """当 pipeline 首步（imputer）期望的列数 > 我们提供的列数时，算出散布索引。

        背景：部分 artifact 的 pipeline 是
            imputer(652) → inf_cleaner → feature_mask(652→515) → scaler(515) → model(515)
        而 artifact 只记录了 mask 之后的 515 列（feature_cols/effective_feature_cols）。
        SimpleImputer 没有 feature_names_in_，**列名无关，位置才重要**，
        所以直接把 515 列按 mask 的 True 位置散回 652 个槽位即可，其余槽位置 NaN
        （NaN 正是 imputer 要处理的东西）。

        返回长度为 len(input_cols) 的索引数组，表示每列应放到第几个槽位；
        无法处理时返回 None。
        """
        expected = ExternalFeatureAugmenter._pipeline_expected_n_features(predictor)
        if expected is None or mask is None:
            return None
        # mask 长度必须等于 pipeline 宽度（如 652），
        # 而 mask 的 True 个数应等于我们手上的列数（如 515）。
        if len(mask) != expected:
            return None
        if int(mask.sum()) != len(input_cols):
            return None
        if expected < len(input_cols):
            return None
        return np.flatnonzero(mask)

    @staticmethod
    def _expand_to_pipeline_width(
        features: pd.DataFrame,
        expander: Optional[np.ndarray],
        expected: Optional[int],
    ) -> pd.DataFrame:
        """把特征矩阵按 expander 索引散布到 pipeline 期望的宽度（缺位填 NaN）。"""
        if expander is None or expected is None or features.shape[1] == expected:
            return features
        arr = features.to_numpy(dtype=float)
        wide = np.full((arr.shape[0], expected), np.nan, dtype=float)
        wide[:, expander] = arr
        return pd.DataFrame(wide, index=features.index)

    @staticmethod
    def _repair_columns_to_length(
        candidate: List[str],
        expected: int,
        audit: Dict[str, Any],
        mask: Optional[Sequence[Any]] = None,
    ) -> Optional[List[str]]:
        """当记录的特征列数与 pipeline 期望不符时，用 feature_mask 反推真正的输入列。

        背景：某些 artifact 的 canonical_feature_cols 会多记录 1 列（重复列/常量列
        处理差异），而 pipeline 的 imputer 是按实际列数 fit 的。此时用 mask 对齐：

            canonical - removed == effective（顺序一致）
            mask 的 True 位置依次对应 effective，False 位置对应 removed

        逐个尝试删掉一个 removed 列，使重建的 mask 与真实 mask 完全一致，
        那一个就是多记录的列。
        """
        removed = [str(c) for c in (audit.get("removed_feature_cols") or [])]
        effective = [str(c) for c in (audit.get("effective_feature_cols") or [])]
        if not removed or not effective or mask is None:
            return None
        mask_list = [bool(m) for m in mask]
        if len(candidate) - 1 != len(mask_list):
            return None

        effective_set = set(effective)
        removed_set = set(removed)
        for drop in removed:
            if drop not in removed_set:
                continue
            built = [c for c in candidate if c != drop]
            if len(built) != len(mask_list):
                continue
            # 重建 mask：True 当且仅当该列在 effective 里
            rebuilt_mask = [c in effective_set and c not in removed_set for c in built]
            if rebuilt_mask == mask_list:
                # 再校验：mask 保留的列依次等于 effective
                kept = [c for c, m in zip(built, mask_list) if m]
                if kept == effective:
                    return built
        return None

    @staticmethod
    def _resolve_input_feature_cols(
        artifact: Dict[str, Any],
        predictor: Any,
        declared_feature_cols: List[str],
    ) -> List[str]:
        """确定模型的真实输入列（可能多于 artifact.feature_cols）。

        背景：Pipeline(imputer → feature_mask → scaler → model) 中，
        imputer 吃全部列（如 2070），mask 后才是模型真正用的列（如 1408），
        而 artifact.feature_cols 常常只记录了 mask 后的 1408 列。
        必须按 imputer 的期望列数喂数据，否则报
        "X has 1408 features, but SimpleImputer is expecting 2070"。

        优先级：
            1. pipeline 各步的 feature_names_in_（最权威，带列名）
            2. feature_audit.canonical_feature_cols（最完整，必要时用 mask 修复长度）
            3. extra.final_feature_names / screening_reference_X 列
            4. artifact.feature_cols（兜底）
        """
        declared = [str(c) for c in (declared_feature_cols or [])]
        extra = artifact.get("extra") or {}
        audit = extra.get("feature_audit") or {}
        expected = ExternalFeatureAugmenter._pipeline_expected_n_features(predictor)

        candidates: List[List[str]] = []

        # 1) pipeline 步的 feature_names_in_（带列名且是 fit 时真实列）
        try:
            for _name, step in (getattr(predictor, "steps", None) or []):
                names = getattr(step, "feature_names_in_", None)
                if names is not None and len(names):
                    candidates.append([str(c) for c in names])
                    break
        except Exception:
            pass

        # 2) canonical_feature_cols
        canonical = audit.get("canonical_feature_cols")
        if isinstance(canonical, (list, tuple)) and canonical:
            candidates.append([str(c) for c in canonical])

        # 3) final_feature_names / screening_reference_X
        finals = extra.get("final_feature_names")
        if isinstance(finals, (list, tuple)) and finals:
            candidates.append([str(c) for c in finals])
        ref = extra.get("screening_reference_X")
        if ref is not None and hasattr(ref, "columns"):
            candidates.append([str(c) for c in ref.columns])

        # 4) 声明的 feature_cols
        if declared:
            candidates.append(declared)

        mask = None
        try:
            for _name, step in (getattr(predictor, "steps", None) or []):
                if hasattr(step, "feature_mask"):
                    mask = list(step.feature_mask)
                    break
        except Exception:
            pass

        for cand in candidates:
            if expected is None or len(cand) == expected:
                if len(cand) > len(declared) or not declared:
                    return cand
                return declared
            if expected is not None and len(cand) > expected:
                repaired = ExternalFeatureAugmenter._repair_columns_to_length(
                    cand, expected, audit, mask
                )
                if repaired:
                    return repaired
        return declared

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
                "has_molecular_workflow": self.has_molecular_workflow(e),
                "molecular_workflow_steps": self.molecular_workflow_step_count(e),
                "depends_on_models": [
                    str(self.entries[j]["name"]) for j in (e.get("depends_on_models") or [])
                ],
                "cascade_inputs": {
                    str(k): str(self.entries[v]["name"])
                    for k, v in (e.get("cascade_inputs") or {}).items()
                },
            }
            for e in self.entries
        ]

    # -- 模型自带分子特征 workflow（关键：必须优先复用）----------------------
    @staticmethod
    def get_molecular_workflow(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """取出模型内部保存的分子特征提取配方（训练时的原始流程）。"""
        extra = entry.get("extra") or {}
        wf = extra.get("molecular_feature_workflow")
        if isinstance(wf, dict) and wf.get("steps"):
            return wf
        return None

    @classmethod
    def has_molecular_workflow(cls, entry: Dict[str, Any]) -> bool:
        return cls.get_molecular_workflow(entry) is not None

    @classmethod
    def molecular_workflow_step_count(cls, entry: Dict[str, Any]) -> int:
        wf = cls.get_molecular_workflow(entry)
        return len(wf.get("steps") or []) if wf else 0

    @classmethod
    def workflow_required_source_columns(cls, entry: Dict[str, Any]) -> List[str]:
        """workflow 需要的全部源列（SMILES / BigSMILES 列）。"""
        wf = cls.get_molecular_workflow(entry)
        if not wf:
            return []
        contract = wf.get("input_contract") or {}
        cols = list(contract.get("selected_source_columns") or [])
        if not cols:
            for step in wf.get("steps") or []:
                for col in step.get("source_columns") or []:
                    if col not in cols:
                        cols.append(col)
        return [str(c) for c in cols]

    @classmethod
    def workflow_output_columns(cls, entry: Dict[str, Any]) -> List[str]:
        """workflow 能产出的全部特征列名。"""
        wf = cls.get_molecular_workflow(entry)
        if not wf:
            return []
        names = [str(c) for c in (wf.get("final_feature_names") or [])]
        if not names:
            for step in wf.get("steps") or []:
                names.extend(str(c) for c in (step.get("feature_names") or []))
        return names

    def replay_molecular_workflow(
        self,
        df: pd.DataFrame,
        *,
        device: Any = None,
        progress_callback: Optional[Callable[[dict], None]] = None,
        fill_missing_source_columns: bool = True,
        skip_unavailable_steps: bool = True,
        only_needed_steps: bool = True,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """回放各模型自带的分子特征 workflow，产出训练时用的那套特征列。

        这是**首选路径**：模型训练时用什么配方提取特征，预测时就用同一配方，
        而不是自己猜 RDKit/Mordred 特征（那会得到不相干的列，且对上千个特征
        逐个试探会卡死）。

        参数:
            fill_missing_source_columns: 缺失的源列（如工作区没有 resin_2_structure）
                                        自动补空列，让 workflow 能跑（那些步骤产出 NaN，
                                        与训练时的行为一致）
            skip_unavailable_steps:     后端不可用（如未安装 xtb）的步骤跳过而非中断
            only_needed_steps:          True（默认）—— 只执行产物被模型 pipeline 真正
                                        需要的步骤。训练时可能试了很多方法（力场、
                                        反应模拟），但最终被 feature_mask 剔除的特征
                                        对预测毫无影响，重算它们纯属浪费（实测环氧反应
                                        模拟 39.8s、力场 26s，占整个回放的 96%）。

        返回:
            (增强后的 df, 每个模型的回放报告)
        """
        try:
            from .molecular_feature_workflow import execute_molecular_feature_workflow
        except ImportError:  # pragma: no cover
            from molecular_feature_workflow import execute_molecular_feature_workflow

        out = df.copy()
        reports: List[Dict[str, Any]] = []

        for entry in self.entries:
            wf = self.get_molecular_workflow(entry)
            report: Dict[str, Any] = {
                "model_name": entry["name"],
                "status": "skipped",
                "reason": None,
                "n_source_columns": 0,
                "filled_source_columns": [],
                "n_output_columns": 0,
                "n_new_columns": 0,
                "skipped_steps": [],
                "warnings": [],
            }
            if wf is None:
                report["reason"] = "模型未保存 molecular_feature_workflow"
                reports.append(report)
                continue

            source_cols = self.workflow_required_source_columns(entry)
            report["n_source_columns"] = len(source_cols)
            work = out
            if fill_missing_source_columns:
                missing = [c for c in source_cols if c not in work.columns]
                if missing:
                    work = work.copy()
                    for col in missing:
                        work[col] = np.nan
                    report["filled_source_columns"] = missing

            still_missing = [c for c in source_cols if c not in work.columns]
            if still_missing:
                report["status"] = "failed"
                report["reason"] = "缺少源列: " + ", ".join(still_missing)
                reports.append(report)
                continue

            # 裁剪 workflow：只保留产物被模型需要的步骤
            run_wf = wf
            if only_needed_steps:
                run_wf, skipped = self._prune_workflow_to_needed_steps(entry, wf)
                report["skipped_steps"] = skipped

            try:
                execution = execute_molecular_feature_workflow(
                    work.reset_index(drop=True),
                    run_wf,
                    device=device,
                    mode="training_import",
                    progress_callback=progress_callback,
                )
            except Exception as exc:
                if not skip_unavailable_steps:
                    report["status"] = "failed"
                    report["reason"] = f"回放异常: {exc}"
                    reports.append(report)
                    continue
                report["status"] = "failed"
                report["reason"] = f"回放异常: {exc}"
                reports.append(report)
                continue

            features = execution.features.reset_index(drop=True)
            features.index = out.index
            # 已存在的同名列先删掉，用 workflow 新算的值覆盖（训练时就是这么算的）
            replace_cols = [c for c in features.columns if c in out.columns]
            if replace_cols:
                out = out.drop(columns=replace_cols)
            out = pd.concat([out, features], axis=1)

            report["status"] = "ok"
            report["n_output_columns"] = int(features.shape[1])
            report["n_new_columns"] = int(len([c for c in features.columns if c not in df.columns]))
            report["warnings"] = [str(w) for w in (execution.warnings or [])][:20]
            report["workflow_hash"] = execution.workflow_hash
            reports.append(report)

        return out, reports

    @classmethod
    def _prune_workflow_to_needed_steps(
        cls, entry: Dict[str, Any], wf: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """裁掉产物完全不被模型需要的步骤。

        判定依据（按可靠性从高到低）：

            1. **workflow.feature_source_map**（最权威）——
               `{特征名: step_id}`，由导出时生成，**直接告诉每个特征是哪步算的**。
               例：`resin_1_structure_xtb_homo → batch_1`。
            2. step.feature_names（若存在）——带/不带 prefix 与需求列比对。

        为什么必须优先用 feature_source_map：实测 `dsc初始温度.joblib` 的 workflow
        有 31 步，但**步骤里根本没有 feature_names 字段**（只有 prefix/source_columns/
        params），旧逻辑 `names` 恒为空 → `not names` 为真 → **全部保留**，导致
        22 个重复 xTB 步骤 + 5 个多余指纹步全部白跑（实测 3 分钟以上）。
        改用 feature_source_map 后：31 步 → **只留 3 步**。

        安全策略：
            - 模型需要列未知时不做任何裁剪（宁慢不错）
            - 所有步骤都被裁掉时退回原 workflow
            - 只裁“产物零命中”的步骤，部分命中的照跑
            - 无 feature_source_map 且无 feature_names 时**不裁**（无法判定）
        """
        # 模型 pipeline 的真实输入列 + artifact 声明的列，取并集（宁可多算不可漏算）
        needed = set(str(c) for c in (entry.get("input_feature_cols") or []))
        needed |= set(str(c) for c in (entry.get("feature_cols") or []))
        extra = entry.get("extra") or {}
        audit = extra.get("feature_audit") or {}
        for key in ("canonical_feature_cols", "effective_feature_cols"):
            vals = audit.get(key)
            if isinstance(vals, (list, tuple)):
                needed |= set(str(c) for c in vals)
        if not needed:
            return wf, []

        steps = list(wf.get("steps") or [])
        if not steps:
            return wf, []

        # ── 主依据：feature_source_map ────────────────────────────────────
        # 注意：workflow 自带的 feature_source_map 优先；entry.extra 里的同名键
        # 可能属于 artifact 而非 workflow，两者都看（workflow 更权威）。
        fsm: Dict[str, Any] = {}
        for src in (wf.get("feature_source_map"), extra.get("feature_source_map")):
            if isinstance(src, dict) and src:
                fsm.update({str(k): v for k, v in src.items()})

        if fsm:
            by_step: Dict[str, List[str]] = {}
            for feat, sid in fsm.items():
                by_step.setdefault(str(sid), []).append(feat)

            # 先找出“有登记产出且命中”的步骤，按 (method, prefix) 建索引。
            # 用途：判定那些“0 登记产出”的步骤是不是重复——
            # 实测 dsc初始温度 里 single_3/single_6 与 single_9 的
            # (method, prefix, source_columns) 完全一致，且 single_9 已产出
            # 全部 167 个被需要的指纹位；single_3/6 是导出时的重复记录。
            def _step_sig(st: Dict[str, Any]) -> Tuple[str, str, str]:
                return (
                    str(st.get("method") or ""),
                    str(st.get("prefix") or ""),
                    "|".join(str(c) for c in (st.get("source_columns") or [])),
                )

            sig_has_real: Dict[Tuple[str, str, str], str] = {}
            # 另建「prefix + method → 已有真产出步骤」索引。
            # 用途：single_4(prefix='resin_', 源 resin_1+2) 与
            # single_7(prefix='resin_', 源 resin_1+2+3) 产出**同名列**
            # （都是 resin_Resin_MACCS_*），只是 source_columns 多少不同；
            # 后者已产出全部需要的列，前者纯属重复。
            # 所以只看 (method, prefix) 就够了，source_columns 不计入。
            sig_prefix_real: Dict[Tuple[str, str], str] = {}
            for st in steps:
                sid = str(st.get("step_id"))
                produced = by_step.get(sid) or []
                if produced and any(f in needed for f in produced):
                    sig_prefix_real.setdefault(
                        (str(st.get("method") or ""), str(st.get("prefix") or "")), sid
                    )

            keep_steps: List[Dict[str, Any]] = []
            skipped: List[Dict[str, Any]] = []
            for step in steps:
                sid = str(step.get("step_id"))
                produced = by_step.get(sid) or []
                hits = [f for f in produced if f in needed]
                if hits:
                    keep_steps.append(step)
                    continue
                if produced:
                    # 产出已知且零命中 → 确定可裁
                    skipped.append({
                        "step_id": step.get("step_id"),
                        "method": step.get("method"),
                        "n_features": len(produced),
                        "reason": "产物不被模型使用（feature_source_map 判定）",
                    })
                    continue
                # ── 产出未知（未登记在 feature_source_map）──
                # 1a) 同签名（method+prefix+源列）已有“真产出”步骤 → 重复，裁
                twin = sig_has_real.get(_step_sig(step))
                if twin and twin != sid:
                    skipped.append({
                        "step_id": step.get("step_id"),
                        "method": step.get("method"),
                        "n_features": 0,
                        "reason": f"与 {twin} 签名相同（重复步骤），产物已由后者提供",
                    })
                    continue
                # 1b) 同 (method, prefix) 已有“真产出”步骤 → 产出同名列，本步重复。
                #     例：single_4 与 single_7 都是 prefix='resin_' 的 MACCS 步，
                #     single_7 已产出全部 231 个 resin_Resin_MACCS_*。
                twin2 = sig_prefix_real.get(
                    (str(step.get("method") or ""), str(step.get("prefix") or ""))
                )
                if twin2 and twin2 != sid:
                    skipped.append({
                        "step_id": step.get("step_id"),
                        "method": step.get("method"),
                        "n_features": 0,
                        "reason": (
                            f"与 {twin2} 同方法同 prefix『{step.get('prefix')}』，"
                            "产出同名列，已由后者提供"
                        ),
                    })
                    continue
                # 2) prefix 前缀 + 方法标志双重校验
                prefix = str(step.get("prefix") or "")
                if prefix:
                    method = str(step.get("method") or "")
                    pre_hits = [n for n in needed if n.startswith(prefix)]
                    if pre_hits and _is_method_output(method, pre_hits):
                        keep_steps.append(step)
                        continue
                    skipped.append({
                        "step_id": step.get("step_id"),
                        "method": step.get("method"),
                        "n_features": 0,
                        "reason": (
                            f"无产物登记且无『{method}』类型的列以 prefix『{prefix}』开头"
                            if pre_hits else
                            f"无产物登记且无列以 prefix『{prefix}』开头"
                        ),
                    })
                    continue
                keep_steps.append(step)
        else:
            # ── 回退依据：step.feature_names ──────────────────────────────
            keep_steps = []
            skipped = []
            any_names = False
            for step in steps:
                names = [str(n) for n in (step.get("feature_names") or [])]
                if names:
                    any_names = True
                prefix = str(step.get("prefix") or "")
                prefixed = [
                    (f"{prefix}_{n}" if prefix and not n.startswith(prefix) else n)
                    for n in names
                ]
                hits = sum(1 for n in (names + prefixed) if n in needed)
                if hits > 0 or not names:
                    keep_steps.append(step)
                else:
                    skipped.append({
                        "step_id": step.get("step_id"),
                        "method": step.get("method"),
                        "n_features": len(names),
                        "reason": "产物不被模型使用（feature_names 判定）",
                    })
            if not any_names:
                # 两条依据都没有 → 无法判定，不裁（宁慢不错）
                return wf, []

        if not keep_steps:
            return wf, []
        if len(keep_steps) == len(steps):
            return wf, []

        pruned = dict(wf)
        pruned["steps"] = keep_steps
        keep_ids = [str(s.get("step_id")) for s in keep_steps]
        merge_order = [sid for sid in (wf.get("merge_order") or []) if str(sid) in keep_ids]
        pruned["merge_order"] = merge_order or keep_ids
        # 同步裁剪 feature_source_map（只留保留步骤产出的特征）
        if fsm:
            pruned["feature_source_map"] = {
                k: v for k, v in fsm.items() if str(v) in set(keep_ids)
            }
        return pruned, skipped

    def required_features(self) -> List[str]:
        """所有模型需要的特征名并集（保持首次出现顺序）。

        用 input_feature_cols（模型 pipeline 的真实输入，可能多于 artifact.feature_cols）。
        """
        seen: List[str] = []
        for entry in self.entries:
            for col in entry.get("input_feature_cols") or entry["feature_cols"]:
                if col not in seen:
                    seen.append(col)
        return seen

    def cascade_external_features(self) -> List[str]:
        """只返回**需要从外部补齐**的特征（排除可由其他模型预测提供的）。

        级联场景下这很关键：若 tg_c 由另一个模型提供，就不该再去总表/提取引擎
        里为它苦苦搜寻（既慢又可能查错），而应留给级联流程。
        """
        cascade_keys = {
            _dep_key(f) for e in self.entries for f in (e.get("cascade_inputs") or {})
        }
        return [f for f in self.required_features() if _dep_key(f) not in cascade_keys]

    def cascade_info(self) -> Dict[str, Any]:
        """级联依赖概览（供 UI 展示）。"""
        graph = self.dependency_graph
        layers = graph.layers()
        return {
            "has_dependency": graph.has_any_dependency(),
            "dependencies": graph.describe(),
            "cycles": [
                [str(self.entries[i]["name"]) for i in L] for L in graph.cycles()
            ],
            "layers": [
                [str(self.entries[i]["name"]) for i in L] for L in layers
            ],
            "external_features": self.cascade_external_features(),
        }

    # -- 特征解析 -----------------------------------------------------------
    def resolve_features(
        self,
        df: pd.DataFrame,
        *,
        manual_overrides: Optional[Dict[str, str]] = None,
        allow_fuzzy: bool = True,
        features: Optional[Sequence[str]] = None,
        cascade_sources: Optional[Dict[str, str]] = None,
    ) -> FeatureResolution:
        """把模型特征名映射到工作区实际列名。

        参数:
            manual_overrides: {外部特征名: 工作区列名}，优先级最高
            allow_fuzzy:      是否启用模糊匹配（关闭则未匹配项直接进 unresolved）
            features:         只解析这些特征（默认解析全部模型的并集）。
                              按模型分别解析可避免"一个模型缺特征连坐其他模型"。
            cascade_sources:  {依赖键: 上游模型预测列名}。级联模型用：
                              若某特征名归一化后命中上游 target，则直接用上游预测列。
        """
        manual_overrides = {str(k): str(v) for k, v in (manual_overrides or {}).items()}
        columns = [str(c) for c in df.columns]
        case_map: Dict[str, str] = {}
        norm_map: Dict[str, str] = {}
        for col in columns:
            case_map.setdefault(col.strip().lower(), col)
            norm_map.setdefault(normalize_name(col), col)

        result = FeatureResolution(
            manual={}, exact={}, case_insensitive={}, normalized={}, alias={}, cascade={},
            pattern={}, fuzzy={}, unresolved=[], pattern_notes={},
        )

        # 级联来源：{依赖键: 已算好的预测列名}
        # 上游模型的预测值已写回 df，可作为下游模型的输入特征。
        cascade_sources = {
            str(k): str(v) for k, v in (cascade_sources or {}).items() if str(v) in columns
        }

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
            # 5.5) 级联：该特征其实是某个已算完模型的预测值。
            #      必须排在真实列匹配（2-5）之后 —— **工作区已有的真实值
            #      一定胜过模型预测值**；又必须排在模式推导/模糊之前，
            #      否则 tg_c 可能被模糊匹配到别的列。
            _ck = _dep_key(feature)
            if _ck and _ck in cascade_sources:
                result["cascade"][feature] = cascade_sources[_ck]
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

    def diagnose_entry(
        self,
        df: pd.DataFrame,
        entry: Dict[str, Any],
        *,
        cascade_sources: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """诊断单个模型能否在当前数据上运行（不预测，供 UI 预览）。"""
        feature_cols = entry["feature_cols"]
        resolution = self.resolve_features(
            df, features=feature_cols, cascade_sources=cascade_sources,
        )
        mapping = resolution.resolved
        pattern_keys = resolution.get("pattern") or {}
        cascade_keys = resolution.get("cascade") or {}
        rows: List[Dict[str, Any]] = []
        for feature in feature_cols:
            if feature in cascade_keys:
                rows.append({
                    "feature": feature,
                    "source": f"<级联: {cascade_keys[feature]}>",
                    "strategy": "cascade",
                })
            elif feature in pattern_keys:
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
        replay_workflow: bool = True,
        enable_cascade: bool = True,
        device: Any = None,
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
            replay_workflow:   True（默认）—— **优先回放模型自带的分子特征 workflow**，
                               产出训练时用的那套特征列。这是关键：模型训练时用什么
                               配方，预测时就用同一配方，而不是自己猜方法。
            enable_cascade:    True（默认）—— **级联模型支持**。若模型 B 的输入特征
                               本身就是模型 A 的预测目标（如 tg_c ↔ 拉伸模量互引），
                               自动按依赖拓扑分层：先算 A、把预测值当 B 的输入。
                               互相依赖的环会自动择优打破（优先用工作区真实值/手工映射）。
                               设为 False 则退回旧行为（各算各的，缺的特征一律 NaN）。
            device:            提取后端设备（如 torch device）

        返回:
            (augmented_df, reports)。reports 首元素为 workflow 回放汇总（若有）；
            级联场景下还会插入一条 `kind='cascade_summary'` 的汇总。
        """
        if output_mode not in ("new_column", "fill_missing", "overwrite"):
            raise ValueError("output_mode 必须是 new_column / fill_missing / overwrite 之一")

        result = df.copy()
        reports: List[Dict[str, Any]] = []
        self._enable_cascade = bool(enable_cascade)

        # ---- 第 0 步（关键）：优先回放模型自带的分子特征 workflow ----
        workflow_reports: List[Dict[str, Any]] = []
        if replay_workflow and any(self.has_molecular_workflow(e) for e in self.entries):
            result, workflow_reports = self.replay_molecular_workflow(
                result, device=device,
            )
            ok = [r for r in workflow_reports if r["status"] == "ok"]
            failed = [r for r in workflow_reports if r["status"] == "failed"]
            total_new = sum(r["n_new_columns"] for r in ok)
            note = (
                f"已回放 {len(ok)}/{len(workflow_reports)} 个模型自带 workflow，"
                f"新增 {total_new} 个特征列"
            )
            if failed:
                note += f"；{len(failed)} 个失败（{'；'.join(str(r.get('reason')) for r in failed[:2])}）"
            reports.append({
                "kind": "molecular_workflow_replay",
                "status": "ok" if ok else "failed",
                "note": note,
                "details": workflow_reports,
            })

        # ---- 级联分层求解 ----
        # 若模型之间存在依赖（B 的输入特征是 A 的预测目标），必须按拓扑顺序
        # 先算 A、把预测值写回 result，再算 B。互相依赖的环（tg_c ↔ 拉伸模量）
        # 无法拓扑排序，此时按“外部可获得性”打破环（见 _break_cycle）。
        graph = self.dependency_graph
        if not enable_cascade or not graph.has_any_dependency():
            scc_layers: List[List[List[int]]] = [[[i]] for i in range(len(self.entries))]
        else:
            scc_layers = graph.scc_layers()
        cascade_sources: Dict[str, str] = {}
        cycle_notes: List[str] = []

        for layer_idx, layer_sccs in enumerate(scc_layers):
            for comp in layer_sccs:
                # 环内（SCC size > 1）：挑一个“最可能被外部解出”的模型先算，
                # 其余等它的预测值。单节点 SCC 就是普通模型。
                ordered = self._order_layer(
                    comp, result, cascade_sources, manual_overrides, allow_fuzzy
                )
                if len(comp) > 1:
                    cycle_notes.append(
                        "互相依赖：" + " ↔ ".join(str(self.entries[i]["name"]) for i in comp)
                        + f"（按 {' → '.join(str(self.entries[i]['name']) for i in ordered)} 顺序求解）"
                    )
                for i in ordered:
                    entry = self.entries[i]
                    result, report = self._predict_one(
                        result, entry, i,
                        output_mode=output_mode,
                        suffix=suffix,
                        add_source_flag=add_source_flag,
                        manual_overrides=manual_overrides,
                        allow_fuzzy=allow_fuzzy,
                        allow_partial=allow_partial,
                        min_feature_coverage=min_feature_coverage,
                        cascade_sources=cascade_sources if enable_cascade else None,
                    )
                    report["layer"] = layer_idx
                    if entry.get("depends_on_models"):
                        report["cascade_depends_on"] = [
                            str(self.entries[j]["name"]) for j in entry["depends_on_models"]
                        ]
                    reports.append(report)

                    # 把刚算出的预测值登记为下游可用的级联来源（仅在级联开启时）
                    # 失败时列不存在或全 NaN，resolve 阶段自然回退到其他策略。
                    if (enable_cascade and report["status"] in ("ok", "noop")
                            and report["output_col"] in result.columns):
                        _tk = _dep_key(entry.get("target_col") or "")
                        if _tk:
                            cascade_sources[_tk] = report["output_col"]
                        # 也登记原始 target 名（若该列确实存在于结果里）
                        _raw = str(entry.get("target_col") or "")
                        if _raw and _raw in result.columns:
                            cascade_sources.setdefault(_dep_key(_raw), _raw)

        if cycle_notes:
            reports.insert(0, {
                "kind": "cascade_summary",
                "status": "ok",
                "note": "；".join(cycle_notes),
                "layers": [[str(self.entries[i]["name"]) for i in L] for L in graph.layers()],
                "cycles": [[str(self.entries[i]["name"]) for i in C] for C in graph.cycles()],
                "dependencies": graph.describe(),
            })

        return result, reports

    # -- 单模型预测（级联分层调用的最小单元）-----------------------------
    def _predict_one(
        self,
        result: pd.DataFrame,
        entry: Dict[str, Any],
        entry_index: int,
        *,
        output_mode: str,
        suffix: str,
        add_source_flag: bool,
        manual_overrides: Optional[Dict[str, str]],
        allow_fuzzy: bool,
        allow_partial: bool,
        min_feature_coverage: float,
        cascade_sources: Optional[Dict[str, str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """对单个模型解析特征 → 构矩阵 → 预测 → 写回。返回 (result, report)。"""
        target = entry["target_col"]
        feature_cols = entry["feature_cols"]
        # 模型 pipeline 的真实输入列（可能多于 feature_cols，如 imputer 吃 2070 列）
        input_cols = list(entry.get("input_feature_cols") or feature_cols)
        out_col = target if output_mode != "new_column" else f"{target}{suffix}"

        # 按模型分别解析，避免"一个模型缺特征连坐其他模型"
        resolution = self.resolve_features(
            result, manual_overrides=manual_overrides, allow_fuzzy=allow_fuzzy,
            features=input_cols, cascade_sources=cascade_sources,
        )
        n_resolved = sum(
            1 for f in input_cols
            if f in resolution.resolved or f in (resolution.get("pattern") or {})
        )
        coverage_ratio = n_resolved / max(1, len(input_cols))
        report: Dict[str, Any] = {
            "name": entry["name"],
            "target_col": target,
            "output_col": out_col,
            "output_mode": output_mode,
            "status": "ok",
            "n_features_required": len(input_cols),
            "n_features_resolved": n_resolved,
            "n_features_model": len(feature_cols),
            "feature_coverage": float(coverage_ratio),
            "unresolved": list(resolution.unresolved),
            "needs_review": dict(resolution.needs_review),
            "from_cascade": dict(resolution.get("cascade") or {}),
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
            return result, report
        if coverage_ratio < min_feature_coverage:
            report["status"] = "skipped"
            report["note"] = (
                f"特征覆盖仅 {coverage_ratio*100:.0f}%（低于阈值 {min_feature_coverage*100:.0f}%），已跳过"
            )
            return result, report

        try:
            features = self.build_feature_frame(result, resolution, entry, feature_cols=input_cols)
        except Exception as exc:
            report["status"] = "error"
            report["note"] = f"构造特征失败: {exc}"
            return result, report

        # 严格按模型 pipeline 的输入契约排序列（含 imputer 需要的全部列）
        features = features.reindex(columns=input_cols)
        # 若 pipeline 首步（imputer）期望的列数 > 我们手上的列数，
        # 按 feature_mask 把列散布到正确槽位，缺位填 NaN（imputer 会处理）。
        features = self._expand_to_pipeline_width(
            features, entry.get("input_expander"), entry.get("pipeline_n_features")
        )
        _notes: List[str] = []
        if resolution.unresolved:
            _notes.append(f"{len(resolution.unresolved)} 个特征缺失已置 NaN，交由模型内置填充处理")
        if report["from_cascade"]:
            _notes.append(
                f"{len(report['from_cascade'])} 个特征来自上游模型预测值（级联）"
            )
        report["note"] = "；".join(_notes)

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
            return result, report

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
            return result, report

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
            else:
                flag.loc[predict_mask] = "predicted"
            result[f"{out_col}_source"] = flag

        report["n_predicted"] = int(predict_mask.sum())
        report["coverage"] = float(pd.Series(out).notna().mean())
        if len(preds):
            report["pred_min"] = float(np.nanmin(preds))
            report["pred_max"] = float(np.nanmax(preds))
            report["pred_mean"] = float(np.nanmean(preds))
        return result, report

    # -- 环处理 -------------------------------------------------------------
    def _order_layer(
        self,
        layer: List[int],
        df: pd.DataFrame,
        cascade_sources: Dict[str, str],
        manual_overrides: Optional[Dict[str, str]],
        allow_fuzzy: bool,
    ) -> List[int]:
        """决定环内模型的求解顺序。

        环（如 tg_c ↔ tensile_modulus_gpa）无法拓扑排序，必须挑一个先算。
        挑法：给每个候选模型打分，**分数高的先算**。关键准则：

            先算“**依赖已被满足**”的那个。

        环内每个模型的依赖都来自环内其他模型，所以依赖不可能被对方先算出来。
        唯一能打破环的，是依赖特征能从**环外**获得：

          + 手工映射到工作区已有列 × 100  —— 用户明确指定，最强信号
          + 依赖特征名已在工作区（真实值）× 50
          + 依赖特征已由环外上游模型算出（级联）× 30
          + 自身可解析特征比例 × 10        —— 平局时的次选依据
          − 自身缺失特征数 × 2

        例：tg_c ↔ tensile_modulus_gpa 互引。
          - 若工作区已有 tg_c 真实值 → 拉伸模量的依赖已满足 → **先算拉伸模量**，
            再用它的预测值算 tg_c。
          - 若两者都无 → 依赖都未满足，退化为比可解析特征比例。
        """
        if len(layer) <= 1:
            return list(layer)

        columns = [str(c) for c in df.columns]
        case_map: Dict[str, str] = {}
        norm_map: Dict[str, str] = {}
        for c in columns:
            case_map.setdefault(c.strip().lower(), c)
            norm_map.setdefault(normalize_name(c), c)
        manual_overrides = manual_overrides or {}
        scored: List[Tuple[float, int]] = []
        for i in layer:
            entry = self.entries[i]
            input_cols = list(entry.get("input_feature_cols") or entry["feature_cols"])
            score = 0.0
            # ★ 核心：环内依赖的“环外可获得性”
            for dep_key in (entry.get("cascade_inputs") or {}):
                # 1) 手工映射到工作区已有列
                _mo = manual_overrides.get(dep_key)
                if _mo and _mo in columns:
                    score += 100.0
                    continue
                # 2) 工作区已有同名/归一化列（真实值）
                if dep_key in columns or case_map.get(dep_key.lower()) or norm_map.get(dep_key):
                    score += 50.0
                    continue
                # 3) 环外上游模型已算出
                if dep_key in cascade_sources:
                    score += 30.0
            # 自身可解析特征比例（平局时的次选依据）
            try:
                res = self.resolve_features(
                    df, manual_overrides=manual_overrides, allow_fuzzy=allow_fuzzy,
                    features=input_cols, cascade_sources=cascade_sources,
                )
                n_ok = sum(1 for f in input_cols if f in res.resolved or f in (res.get("pattern") or {}))
                score += 10.0 * (n_ok / max(1, len(input_cols)))
                score -= 2.0 * len(res.unresolved)
            except Exception:
                pass
            scored.append((score, i))
        # 分数降序；同分按原始顺序（稳定）
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [i for _s, i in scored]



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
    enable_cascade: bool = True,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """一次性用多个外部模型补齐特征列（通用）。

    enable_cascade=True（默认）时，自动识别模型间的特征依赖（B 的输入特征
    是 A 的预测目标）并按拓扑顺序求解，包括处理互相引用的环。
    """
    augmenter = ExternalFeatureAugmenter(model_blobs, model_names=model_names, alias_table=alias_table)
    return augmenter.augment(
        df,
        manual_overrides=manual_overrides,
        output_mode=output_mode,
        suffix=suffix,
        add_source_flag=add_source_flag,
        enable_cascade=enable_cascade,
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


def describe_cascade(model_blobs: Sequence[bytes]) -> Dict[str, Any]:
    """只分析模型间的级联依赖（不预测），用于 UI 预览。

    返回 {has_dependency, dependencies, cycles, layers, external_features}。
    """
    return ExternalFeatureAugmenter(model_blobs).cascade_info()
