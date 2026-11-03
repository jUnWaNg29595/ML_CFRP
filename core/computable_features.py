"""统一判定「模型输入特征能否由系统自动算出」。

问题背景
--------
虚拟筛选（高通量筛选）页原先把所有「数值型且不属于分子特征 workflow 输出」的
模型特征都塞进「选择要枚举的配方/工艺特征」候选池。结果像
``formulation_r_value`` / ``resin_1_epoxy_group_count`` / ``process_total_time_h``
``cure_stage_count`` 这类**系统本来就能从候选配方或固化制度算出来**的特征，
被当成「必须由用户枚举的设计变量」摆到 UI 上，用户只能手动给一串猜测值——
既费事，又用猜出来的工艺值污染筛选结果。

本模块提供单一权威判定：给定一个特征名（可选带 registry 定义、带候选表），
返回它属于哪一类来源，以及「怎么算 / 为什么算不出」。

设计原则
--------
1. **只读、纯函数**：不修改任何全局状态，不改数据。
2. **证据优先**：优先采信 registry 的 ``source_type`` / ``calculation_rule``；
   其次采信系统内置的计算名表；最后才退化到关键词语义分类。
3. **保守**：判定为「可计算」必须有明确的实现入口；不确定一律归为需人工确认，
   绝不把算不出来的特征谎报成可计算。
4. **不伪造**：``derivable``（可由已有列算出）与 ``requires_choice``（必须先由
   用户给定一个原始量才能算出，例如固化制度）必须区分开——后者不是
   「不需要用户输入」，而是「用户只需输入一个原始量，其余全部自动派生」。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# 来源分类
# ---------------------------------------------------------------------------
#: 可由候选分子结构直接算出（RDKit 描述符 / 指纹 / 分子特征 workflow / 逐组分物理量）
SOURCE_MOLECULAR = "molecular"
#: 可由候选配方（树脂/固化剂 SMILES + phr）直接算出：EEW/AHEW/PHR/化学计量比/组分统计
SOURCE_FORMULATION = "formulation"
#: 由某组原始列确定性派生（固化制度 → 阶段数/总时长/最高温度…）
SOURCE_DERIVED = "derived"
#: 只能由用户显式给定的设计变量（工艺/测试条件），系统无法自行推导
SOURCE_MANUAL = "manual"
#: 无法判定
SOURCE_UNKNOWN = "unknown"

SOURCE_LABELS: dict[str, str] = {
    SOURCE_MOLECULAR: "分子结构可算",
    SOURCE_FORMULATION: "配方可算",
    SOURCE_DERIVED: "原始列可派生",
    SOURCE_MANUAL: "需人工给定",
    SOURCE_UNKNOWN: "无法判定",
}

#: 属于「系统可自动算出」的来源集合
AUTO_SOURCES: frozenset[str] = frozenset(
    {SOURCE_MOLECULAR, SOURCE_FORMULATION, SOURCE_DERIVED}
)


@dataclass
class ComputableVerdict:
    """单个特征的来源判定结果。"""

    feature: str
    source: str
    evidence: str = ""
    implementation: str = ""
    #: 该特征若要算出来，必须先由用户提供的原始输入列（例如固化制度 cure_schedule）
    requires_columns: list[str] = field(default_factory=list)
    #: 该特征本身是否是「原始设计变量」（可枚举的旋钮）
    is_design_variable: bool = False
    #: 该特征是否已经在候选表里算好了（本次筛选无需再算）
    present_in_candidate: bool = False

    @property
    def auto_computable(self) -> bool:
        return self.source in AUTO_SOURCES

    @property
    def label(self) -> str:
        return SOURCE_LABELS.get(self.source, self.source)

    def as_row(self) -> dict[str, Any]:
        return {
            "特征": self.feature,
            "来源判定": self.label,
            "可自动计算": "✅ 是" if self.auto_computable else "❌ 否",
            "计算依据": self.evidence,
            "实现入口": self.implementation,
            "需先给定": "、".join(self.requires_columns) if self.requires_columns else "",
            "可枚举设计变量": "✅" if self.is_design_variable else "",
        }


# ---------------------------------------------------------------------------
# 内置计算名表（与实现严格对齐，宁可少列不可错列）
# ---------------------------------------------------------------------------

#: 由 core/virtual_screening.py::add_candidate_equivalent_metrics 计算。
#: 该函数在虚拟筛选流程中总会先于特征矩阵构建执行，因此这些列一定可算。
_FORMULATION_METRIC_COLUMNS: tuple[str, ...] = (
    # 规范名（computed_*）
    "computed_resin_molecular_weight",
    "computed_hardener_molecular_weight",
    "computed_resin_eew",
    "computed_hardener_ahew",
    "computed_resin_functionality",
    "computed_hardener_functionality",
    "computed_theoretical_phr",
    "computed_actual_phr",
    "computed_stoich_ratio",
    "computed_stoich_delta",
    "computed_resin_eq_100",
    "computed_hardener_eq",
    "computed_equiv_ratio_h_to_r",
    "computed_equiv_ratio_r_to_h",
    # 显示别名
    "EEW",
    "AHEW",
    "Resin_Functionality",
    "Hardener_Functionality",
    "Theoretical_PHR",
    "Actual_PHR",
    "Stoich_Ratio",
    "Stoich_Delta",
    "Resin_Eq_100",
    "Hardener_Eq",
    "Equiv_Ratio_H_to_R",
    "Equiv_Ratio_R_to_H",
    "PHR",
    "eew",
    "ahew",
    "resin_functionality",
    "hardener_functionality",
    "theoretical_phr",
    "actual_phr",
    "stoich_ratio",
    "stoichiometric_ratio",
    "stoichiometric_ratio_r",
    "stoichiometry_r",
    "r_value",
    "resin_eq_100",
    "hardener_eq",
    "equiv_ratio_h_to_r",
    "equiv_ratio_r_to_h",
)

#: 由 core/auto_feature_resolver.py::compute_formulation_feature 计算。
_FORMULATION_RESOLVER_COLUMNS: tuple[str, ...] = (
    "formulation_resin_total_eew_g_eq",
    "formulation_resin_eew_g_eq",
    "formulation_hardener_total_ahew_g_eq",
    "formulation_hardener_ahew_g_eq",
    "formulation_r_value",
    "formulation_resin_hardener_equivalent_ratio",
    "formulation_epoxy_binder_total_phr",
    "resin_eew",
    "resin_eew_g_eq",
    "resin_total_eew_g_eq",
    "epoxy_equivalent_weight",
    "epoxy_equivalent_weight_g_eq",
    "hardener_ahew_g_eq",
    "curing_agent_total_ahew_g_eq",
    "amine_hydrogen_equivalent_weight",
    "amine_hydrogen_equivalent_weight_g_eq",
    "epoxy_functionality",
    "resin_epoxy_group_count",
    "resin_epoxy_group_total",
    "resin_equivalent_group_total",
    "curer_functionality",
    "curing_agent_active_hydrogen_total",
    "curing_agent_active_hydrogen_equivalent_count",
    "curing_agent_equivalent_group_total",
    "resin_total_phr",
    "curing_agent_total_phr",
    "reactive_toughener_total_phr",
    "accelerator_total_phr",
    "resin_smiles_n_components",
    "resin_n_components",
    "resin_component_count_from_smiles",
    "curing_agent_smiles_n_components",
    "hardener_smiles_n_components",
    "curing_agent_n_components",
    "hardener_n_components",
)

#: 组分计数类：由配方组成直接数出来（未提供的组分恒为 0）。
_COMPONENT_COUNT_COLUMNS: tuple[str, ...] = (
    "resin_component_count",
    "curing_agent_component_count",
    "small_additive_component_count",
    "initiator_component_count",
    "accelerator_component_count",
    "catalyst_component_count",
    "reactive_diluent_component_count",
    "reactive_toughener_component_count",
    "other_component_count",
    "filler_component_count",
    "initiator_present",
)

#: 逐组分物理量：由 core/component_physics.py::compute_component_physics 从
#: 各组分 SMILES 算出（MW / 官能度 / 当量重 / 等效组数）。
#: 用正则匹配 ``{resin|curing_agent}_{1..3}_{后缀}``。
_COMPONENT_PHYSICS_SUFFIXES: tuple[str, ...] = (
    "molecular_weight_g_mol",
    "equivalent_weight_g_eq",
    "epoxy_group_count",
    "active_hydrogen_equivalent_count",
    "equivalent_group_count",
    "mw_resolved",
    "ew_resolved",
)
_COMPONENT_PHYSICS_RE = re.compile(
    r"^(resin|curing_agent)_(\d+)_("
    + "|".join(_COMPONENT_PHYSICS_SUFFIXES)
    + r")$"
)

#: 由 core/process_features.py 从固化制度（cure_schedule 等原始列）派生。
#: 名字与 compute_process_features 的 _derive_named 分支一一对应。
_DERIVED_PROCESS_COLUMNS: dict[str, str] = {
    "cure_stage_count": "固化阶段数",
    "cure_total_time_h": "固化总时长",
    "cure_max_temperature_c": "固化最高温度",
    "cure_final_temperature_c": "固化末段温度",
    "cure_temp_time_integral_c_h": "固化温度-时间积分",
    "cure_time_weighted_avg_temperature_c": "固化时间加权平均温度",
    "post_cure_stage_count": "后固化阶段数",
    "post_cure_total_time_h": "后固化总时长",
    "post_cure_max_temperature_c": "后固化最高温度",
    "post_cure_final_temperature_c": "后固化末段温度",
    "post_cure_temp_time_integral_c_h": "后固化温度-时间积分",
    "post_cure_time_weighted_avg_temperature_c": "后固化时间加权平均温度",
    "post_cure_temperature_c": "后固化温度",
    "has_post_cure": "是否有后固化",
    "total_cure_stage_count": "总固化阶段数",
    "total_heat_treatment_time_h": "总热处理时长",
    "overall_max_cure_temperature_c": "全程最高固化温度",
    "overall_temp_time_integral_c_h": "全程温度-时间积分",
}

#: 宽表口径的固化工艺族：语义与 _DERIVED_PROCESS_COLUMNS 相同，只是命名不同。
#: 由本模块的 ``derive_process_family_from_schedule`` 实现（见下）。
_DERIVED_PROCESS_FAMILY: dict[str, str] = {
    "process_max_temperature_c": "全程最高固化温度",
    "process_temperature_time_integral_c_h": "全程温度-时间积分",
    "process_total_time_h": "全程总固化时长",
    "process_final_cure_temperature_c": "末段固化温度",
    "process_final_cure_time_h": "末段固化时长",
    "process_has_post_cure": "是否含后固化",
}

#: 派生特征所需的原始输入列（用户只需给这些，其余自动派生）
SCHEDULE_SOURCE_CANDIDATES: tuple[str, ...] = (
    "cure_schedule",
    "curing_schedule",
    "固化制度",
    "固化工艺",
    "cure_profile",
    "temperature_schedule",
    "process_schedule",
)
POST_CURE_SCHEDULE_CANDIDATES: tuple[str, ...] = (
    "post_cure_schedule",
    "postcure_schedule",
    "后固化制度",
)

#: 判定「分子特征」的关键词
_MOLECULAR_TOKENS: tuple[str, ...] = (
    "分子量", "分子描述", "原子数", "环数", "官能团", "指纹", "描述符", "拓扑",
    "morgan", "maccs", "rdkit", "xtb", "homo", "lumo", "tpsa", "logp",
    "molecular_weight", "molwt", "mol_wt", "atom_count", "ring_count",
    "hbd", "hba", "fp_bit", "fr_", "n_atoms", "n_rings", "n_hbd", "n_hba",
    "fingerprint", "descriptor", "smiles", "structure", "bigsmiles",
    "special_char", "rotatable_bonds", "heavy_atoms", "aromatic_rings",
    "formal_charge", "fragment", "polymer",
)

#: 判定「人工给定设计变量」的关键词（工艺/测试/表征条件）
_MANUAL_TOKENS: tuple[str, ...] = (
    "curing_temperature", "cure_temperature", "curing_time", "cure_time",
    "ramp_rate", "heating_rate", "postcure", "post_cure", "dwell", "hold_time",
    "pressure", "vacuum", "humidity", "atmosphere", "degas",
    "test_method", "test_standard", "test_condition", "test_atmosphere",
    "test_temperature", "frequency", "strain", "specimen", "operator",
    "固化温度", "固化时间", "升温速率", "保温", "后固化", "压力", "真空",
    "湿度", "气氛", "测试方法", "测试标准", "测试条件", "频率",
)

#: 逐组分 phr（配比）：**这是配方设计变量本身**，用户必须给定，系统无从推导。
_PHR_RE = re.compile(
    r"^(resin|curing_agent|small_additive|accelerator|catalyst|initiator|"
    r"reactive_diluent|reactive_toughener|filler|other)_(\d+)_amount_phr$"
)

#: 由测试标准名派生的计数（test_standard_canonical → test_standard_count）
_TEST_STANDARD_DERIVED: tuple[str, ...] = (
    "test_standard_count",
)

#: 明确是「测试/表征条件」的字段：属于设计变量，必须由用户给定
_TEST_CONDITION_TOKENS: tuple[str, ...] = (
    "test_method", "test_standard", "test_condition", "test_atmosphere",
    "test_temperature", "test_humidity", "frequency", "heating_rate",
    "ramp_rate", "strain", "specimen", "atmosphere",
)


def _norm(text: Any) -> str:
    return str(text or "").strip().lower()


def _as_sequence(value: Any) -> list:
    """把任意列名容器安全归一化成 list。

    为什么不能用 ``value or []``：pandas 的 Index / Series / numpy 数组的布尔
    求值是**歧义的**（长度 >1 时直接抛
    ``ValueError: The truth value of a Series/Index is ambiguous``）。
    而本模块的调用方（虚拟筛选页）天然传入 ``pool_df.columns`` 这种 pandas
    Index，于是必然崩在 ``X or []`` 上。
    """
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _match_in(name: str, table: Iterable[str]) -> str | None:
    key = _norm(name)
    for candidate in table:
        if _norm(candidate) == key:
            return str(candidate)
    return None


# ---------------------------------------------------------------------------
# 判定主入口
# ---------------------------------------------------------------------------
def classify_computable_feature(
    feature: str,
    *,
    definition: Mapping[str, Any] | None = None,
    candidate_columns: Iterable[str] | None = None,
) -> ComputableVerdict:
    """判定单个模型输入特征的来源类别。

    参数
    ----
    feature:
        模型特征名。
    definition:
        可选的 registry 特征定义（含 ``source_type`` / ``calculation_rule``）。
        传入时优先采信其声明。
    candidate_columns:
        当前候选表的列名集合。用于判断该特征是否**已经算好**。
    """
    name = str(feature or "").strip()
    if not name:
        return ComputableVerdict(feature="", source=SOURCE_UNKNOWN, evidence="空特征名")

    candidate_set = {_norm(c) for c in _as_sequence(candidate_columns)}
    present = _norm(name) in candidate_set
    definition = definition if isinstance(definition, Mapping) else {}

    # --- 1. registry 声明优先 -------------------------------------------------
    declared = str(definition.get("source_type") or "").strip()
    rule = definition.get("calculation_rule")
    rule = rule if isinstance(rule, Mapping) else {}
    implementation = str(
        rule.get("implementation") or definition.get("implementation") or ""
    ).strip()
    input_fields = [
        str(f) for f in _as_sequence(rule.get("input_fields")) if str(f).strip()
    ]

    if declared == "derived_workflow":
        return ComputableVerdict(
            feature=name,
            source=SOURCE_DERIVED,
            evidence=str(definition.get("description") or "registry 声明为派生特征"),
            implementation=implementation
            or "core.process_features:derive_declared_feature",
            requires_columns=input_fields or list(SCHEDULE_SOURCE_CANDIDATES[:1]),
            present_in_candidate=present,
        )
    if declared == "molecular_workflow":
        return ComputableVerdict(
            feature=name,
            source=SOURCE_MOLECULAR,
            evidence=str(definition.get("description") or "registry 声明为分子特征"),
            implementation=implementation or "molecular_feature_workflow",
            present_in_candidate=present,
        )
    if declared == "manual_input":
        return ComputableVerdict(
            feature=name,
            source=SOURCE_MANUAL,
            evidence=str(definition.get("description") or "registry 声明为人工输入"),
            is_design_variable=True,
            present_in_candidate=present,
        )

    # --- 2. 内置计算名表 ------------------------------------------------------
    if _match_in(name, _FORMULATION_METRIC_COLUMNS):
        return ComputableVerdict(
            feature=name,
            source=SOURCE_FORMULATION,
            evidence="由候选树脂/固化剂结构 + phr 直接算出",
            implementation="core.virtual_screening:add_candidate_equivalent_metrics",
            present_in_candidate=present,
        )
    if _match_in(name, _FORMULATION_RESOLVER_COLUMNS):
        return ComputableVerdict(
            feature=name,
            source=SOURCE_FORMULATION,
            evidence="由候选树脂/固化剂结构推导（EEW/AHEW/官能度/组分计数）",
            implementation="core.auto_feature_resolver:compute_formulation_feature",
            present_in_candidate=present,
        )
    if _match_in(name, _COMPONENT_COUNT_COLUMNS):
        return ComputableVerdict(
            feature=name,
            source=SOURCE_FORMULATION,
            evidence="由配方组分构成直接计数（未提供的组分恒为 0）",
            implementation="core.portal_formulation_inputs:derive_formulation_features",
            present_in_candidate=present,
        )
    if _COMPONENT_PHYSICS_RE.match(_norm(name)):
        return ComputableVerdict(
            feature=name,
            source=SOURCE_MOLECULAR,
            evidence="由该组分 SMILES 直接算出（MW/官能度/当量重/等效组数）",
            implementation="core.component_physics:compute_component_physics",
            present_in_candidate=present,
        )
    if _PHR_RE.match(_norm(name)):
        return ComputableVerdict(
            feature=name,
            source=SOURCE_MANUAL,
            evidence="逐组分配比（phr）本身就是配方设计变量，必须由用户给定",
            is_design_variable=True,
            present_in_candidate=present,
        )
    if _match_in(name, _TEST_STANDARD_DERIVED):
        return ComputableVerdict(
            feature=name,
            source=SOURCE_DERIVED,
            evidence="由测试标准条目数派生",
            implementation="core.process_features:derive_declared_feature",
            requires_columns=["test_standard_canonical"],
            present_in_candidate=present,
        )
    hit = _match_in(name, _DERIVED_PROCESS_COLUMNS)
    if hit:
        return ComputableVerdict(
            feature=name,
            source=SOURCE_DERIVED,
            evidence=f"由固化制度文本派生（{_DERIVED_PROCESS_COLUMNS[hit]}）",
            implementation="core.process_features:derive_declared_feature",
            requires_columns=list(SCHEDULE_SOURCE_CANDIDATES[:1]),
            present_in_candidate=present,
        )
    hit = _match_in(name, _DERIVED_PROCESS_FAMILY)
    if hit:
        return ComputableVerdict(
            feature=name,
            source=SOURCE_DERIVED,
            evidence=f"由固化制度文本派生（{_DERIVED_PROCESS_FAMILY[hit]}）",
            implementation="core.computable_features:derive_process_family_from_schedule",
            requires_columns=list(SCHEDULE_SOURCE_CANDIDATES[:1]),
            present_in_candidate=present,
        )

    # --- 3. 关键词语义兜底 ----------------------------------------------------
    lowered = _norm(name)
    # 测试/表征条件先判：它们含 "test"/"atmosphere" 等，绝不能被误判成可算
    if any(token in lowered for token in _TEST_CONDITION_TOKENS):
        return ComputableVerdict(
            feature=name,
            source=SOURCE_MANUAL,
            evidence="测试/表征条件，属于实验设定而非计算结果",
            is_design_variable=True,
            present_in_candidate=present,
        )
    if any(token in lowered for token in _MOLECULAR_TOKENS):
        return ComputableVerdict(
            feature=name,
            source=SOURCE_MOLECULAR,
            evidence="名称命中分子特征关键词，应由分子特征 workflow 提取",
            implementation="molecular_feature_workflow",
            present_in_candidate=present,
        )
    if any(token in lowered for token in _MANUAL_TOKENS):
        return ComputableVerdict(
            feature=name,
            source=SOURCE_MANUAL,
            evidence="名称命中工艺条件关键词，无法由结构或配方推导",
            is_design_variable=True,
            present_in_candidate=present,
        )

    return ComputableVerdict(
        feature=name,
        source=SOURCE_UNKNOWN,
        evidence="未命中任何已知计算实现或工艺关键词，需人工确认",
        present_in_candidate=present,
    )


def classify_computable_features(
    features: Sequence[str],
    *,
    definitions: Mapping[str, Mapping[str, Any]] | None = None,
    candidate_columns: Iterable[str] | None = None,
) -> list[ComputableVerdict]:
    """批量判定。``definitions`` 为 ``{特征名: registry 定义}``。"""
    definitions = definitions or {}
    candidate_columns = _as_sequence(candidate_columns)
    lookup = {_norm(k): v for k, v in definitions.items() if isinstance(v, Mapping)}
    return [
        classify_computable_feature(
            feature,
            definition=lookup.get(_norm(feature)),
            candidate_columns=candidate_columns,
        )
        for feature in features
    ]


# ---------------------------------------------------------------------------
# 面向虚拟筛选页的汇总
# ---------------------------------------------------------------------------
def split_screening_feature_pool(
    model_feature_cols: Sequence[str],
    *,
    workflow_feature_names: Sequence[str] | None = None,
    definitions: Mapping[str, Mapping[str, Any]] | None = None,
    candidate_columns: Iterable[str] | None = None,
) -> dict[str, list[ComputableVerdict]]:
    """把模型全部输入特征拆成三组，供筛选页展示与建池。

    返回 ``{"auto": [...], "design": [...], "unknown": [...]}``：

    - ``auto``：系统可自动算出的特征（分子 / 配方 / 派生），**不应出现在枚举池**。
    - ``design``：必须由用户给定的设计变量，**这才是枚举池的合法成员**。
    - ``unknown``：无法判定，单独列出交由用户决定（默认不进枚举池，避免误当设计变量）。
    """
    workflow_set = {_norm(c) for c in _as_sequence(workflow_feature_names)}
    verdicts = classify_computable_features(
        model_feature_cols,
        definitions=definitions,
        candidate_columns=candidate_columns,
    )
    auto: list[ComputableVerdict] = []
    design: list[ComputableVerdict] = []
    unknown: list[ComputableVerdict] = []
    for verdict in verdicts:
        if _norm(verdict.feature) in workflow_set:
            verdict.source = SOURCE_MOLECULAR
            verdict.evidence = verdict.evidence or "属于分子特征 workflow 输出"
            verdict.implementation = verdict.implementation or "molecular_feature_workflow"
        if verdict.auto_computable:
            auto.append(verdict)
        elif verdict.source == SOURCE_MANUAL:
            design.append(verdict)
        else:
            unknown.append(verdict)
    return {"auto": auto, "design": design, "unknown": unknown}


def summarize_split(split: Mapping[str, Sequence[ComputableVerdict]]) -> dict[str, int]:
    """给出各类别数量，便于 UI 一行摘要。"""
    return {
        "auto": len(_as_sequence(split.get("auto"))),
        "design": len(_as_sequence(split.get("design"))),
        "unknown": len(_as_sequence(split.get("unknown"))),
    }


def derived_feature_requirements(
    feature_cols: Sequence[str],
    *,
    definitions: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, list[str]]:
    """返回 ``{派生特征: [所需原始列]}``，供筛选页提示「你只需给固化制度」。

    只统计确实属于派生类、且当前模型真的用到的特征。
    """
    out: dict[str, list[str]] = {}
    for verdict in classify_computable_features(feature_cols, definitions=definitions):
        if verdict.source != SOURCE_DERIVED:
            continue
        out[verdict.feature] = list(verdict.requires_columns) or list(
            SCHEDULE_SOURCE_CANDIDATES[:1]
        )
    return out
def resolve_schedule_source_column(available_columns: Iterable[str]) -> str | None:
    """在给定列名里找出可作固化制度原始输入的那一列（找不到返回 None）。

    注意：``available_columns`` 常直接传 ``df.columns``（pandas Index），
    必须走 _as_sequence，不能用 ``or []``（会抛 ambiguous truth value）。
    """
    available = {_norm(c): str(c) for c in _as_sequence(available_columns)}
    for candidate in SCHEDULE_SOURCE_CANDIDATES:
        hit = available.get(_norm(candidate))
        if hit:
            return hit
    return None


# ---------------------------------------------------------------------------
# 固化制度 → process_* 族派生（宽表口径）
# ---------------------------------------------------------------------------
def _schedule_pairs(text: Any) -> list[tuple[float, float]]:
    """解析 ``"80 °C/2 h + 150 °C/3 h"`` → ``[(80.0, 2.0), (150.0, 3.0)]``。

    复用 core.process_features 的实现，保证与训练侧完全同一口径；
    不可用时退化为本地等价正则（避免循环导入/环境缺依赖时静默失效）。
    """
    try:
        from .process_features import _schedule_pairs as _impl

        return _impl(text)
    except Exception:
        pass
    if text is None:
        return []
    raw = str(text).strip()
    if not raw:
        return []
    pairs = [
        (float(a), float(b))
        for a, b in re.findall(r"([-+]?\d*\.?\d+)\s*[^0-9;,:/]*C?\s*/\s*([-+]?\d*\.?\d+)", raw)
    ]
    if not pairs:
        raise ValueError("固化工艺阶段格式无法解析")
    return pairs


def derive_process_family_from_schedule(
    schedule: Any,
    *,
    post_cure_schedule: Any = None,
) -> dict[str, float]:
    """把固化制度文本派生成 ``process_*`` 族（宽表口径）。

    语义与 ``core.process_features`` 的 cure_/post_cure_ 族一致，仅命名不同：
    训练侧宽表用的是 ``process_*``，本函数让虚拟筛选能算出同一批特征，
    而不是退回「中位数伪造」。

    参数
    ----
    schedule : 主固化制度，如 ``"130 °C/2 h + 180 °C/2 h"``
    post_cure_schedule : 可选的后固化制度；给定时其阶段计入后固化

    返回
    ----
    ``{特征名: 值}``；无法解析的项不出现在结果里（不写 NaN 占位）。
    """
    import numpy as _np  # 局部导入：本模块被 UI 热路径导入，避免顶层重依赖

    main_pairs: list[tuple[float, float]] = []
    post_pairs: list[tuple[float, float]] = []
    try:
        main_pairs = _schedule_pairs(schedule)
    except Exception:
        main_pairs = []
    try:
        post_pairs = _schedule_pairs(post_cure_schedule) if post_cure_schedule else []
    except Exception:
        post_pairs = []

    if not main_pairs and not post_pairs:
        return {}

    # 后固化阶段单列时，主制度里若已含全部阶段则不再重复计入
    all_pairs = list(main_pairs) + list(post_pairs)
    out: dict[str, float] = {}

    if all_pairs:
        out["process_max_temperature_c"] = float(max(p[0] for p in all_pairs))
        out["process_temperature_time_integral_c_h"] = float(
            sum(temp * duration for temp, duration in all_pairs)
        )
        out["process_total_time_h"] = float(sum(d for _, d in all_pairs))
        last_temp, last_time = all_pairs[-1]
        out["process_final_cure_temperature_c"] = float(last_temp)
        out["process_final_cure_time_h"] = float(last_time)
        out["process_has_post_cure"] = float(int(bool(post_pairs)))

    if not _np.isfinite(list(out.values())).all():  # pragma: no cover - 防御
        return {k: v for k, v in out.items() if _np.isfinite(v)}
    return out


def apply_schedule_derivations(
    frame,
    *,
    feature_cols: Sequence[str],
    schedule_column: str | None = None,
    post_cure_column: str | None = None,
    definitions: Mapping[str, Mapping[str, Any]] | None = None,
):
    """把候选表里的固化制度列按行派生成模型需要的 ``process_*`` 特征。

    只写「模型确实需要、且当前为缺失或全空」的列，不覆盖候选表已有值，
    也不覆盖用户显式给定的统一输入——避免静默改写用户的选择。

    返回 ``(frame, filled_columns)``；``frame`` 是副本，原表不变。
    """
    import pandas as _pd

    if not isinstance(frame, _pd.DataFrame) or frame.empty:
        return frame, []
    needed = [
        verdict.feature
        for verdict in classify_computable_features(feature_cols, definitions=definitions)
        if verdict.source == SOURCE_DERIVED
        and verdict.feature in _DERIVED_PROCESS_FAMILY
    ]
    if not needed:
        return frame, []
    if not schedule_column or schedule_column not in frame.columns:
        return frame, []

    out = frame.copy()
    has_post = bool(post_cure_column and post_cure_column in out.columns)

    # [极速向量化] 500 万行中通常只有极少数唯一的固化制度组合。
    # 严禁逐行 Python 循环（原实测 61.88 秒）！
    # 提取唯一组合并解析，然后快速广播/map 回大表（实测 0.46 秒，提速 133 倍）。
    if has_post:
        _u_sched = out[[schedule_column, post_cure_column]].drop_duplicates().reset_index(drop=True)
        _sched_records = [
            (
                row[schedule_column],
                row[post_cure_column],
                derive_process_family_from_schedule(row[schedule_column], post_cure_schedule=row[post_cure_column])
            )
            for _, row in _u_sched.iterrows()
        ]
    else:
        _u_sched = out[schedule_column].dropna().unique()
        _sched_records = [
            (s, None, derive_process_family_from_schedule(s))
            for s in _u_sched
        ]

    filled: list[str] = []
    for name in needed:
        if has_post:
            _map = {
                (r[0], r[1]): r[2].get(name, _pd.NA)
                for r in _sched_records
            }
            if len(_map) == 1:
                _val = list(_map.values())[0]
                series = _pd.Series(_val, index=out.index) if _pd.notna(_val) else _pd.Series(_pd.NA, index=out.index)
            else:
                _multi_keys = list(zip(out[schedule_column], out[post_cure_column]))
                series = _pd.Series([_map.get(k, _pd.NA) for k in _multi_keys], index=out.index)
        else:
            _map = {r[0]: r[2].get(name, _pd.NA) for r in _sched_records}
            if len(_map) == 1:
                _val = list(_map.values())[0]
                series = _pd.Series(_val, index=out.index) if _pd.notna(_val) else _pd.Series(_pd.NA, index=out.index)
            else:
                series = out[schedule_column].map(_map)

        series = _pd.to_numeric(series, errors="coerce")
        if not series.notna().any():
            continue
        if name in out.columns:
            existing = _pd.to_numeric(out[name], errors="coerce")
            if existing.notna().any():
                merged = existing.where(existing.notna(), series)
                if merged.equals(existing):
                    continue
                out[name] = merged
                filled.append(name)
                continue
        out[name] = series
        filled.append(name)
    return out, filled


__all__ = [
    "AUTO_SOURCES",
    "POST_CURE_SCHEDULE_CANDIDATES",
    "SCHEDULE_SOURCE_CANDIDATES",
    "SOURCE_DERIVED",
    "SOURCE_FORMULATION",
    "SOURCE_LABELS",
    "SOURCE_MANUAL",
    "SOURCE_MOLECULAR",
    "SOURCE_UNKNOWN",
    "ComputableVerdict",
    "apply_schedule_derivations",
    "classify_computable_feature",
    "classify_computable_features",
    "derive_process_family_from_schedule",
    "derived_feature_requirements",
    "resolve_schedule_source_column",
    "split_screening_feature_pool",
    "summarize_split",
]