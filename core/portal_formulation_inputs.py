"""配方自动推导：从 SMILES + phr 推出 EEW / AHEW / 化学计量比 r 等特征。

设计依据：docs/superpowers/specs/2026-09-22-portal-formulation-first-input-design.md §4.3

为什么需要本模块
----------------
门户的 36 个非 workflow 特征里，有 23 个可由配方直接推导（EEW、AHEW、r、
组分计数、官能团汇总）。要求用户手填这些既繁琐又易错——它们本就是配方的
数学后果，不是独立实验变量。

核心公式（已用全表验证）
------------------------
    r = (固化剂 phr / AHEW) / (树脂 phr / EEW)

验证结果（ml_qspr_selected.csv，n=3237）：
    - 本公式与全表 formulation_r_value 相关系数 0.9749，中位绝对误差 0.00017
    - cp_r_value（当量重比 AHEW/EEW）相关系数仅 -0.0886  ← 语义完全不同

⚠️ 禁止使用 cp_r_value 填充 formulation_r_value
------------------------------------------------
``cp_r_value``（core/component_physics.py）的兜底计算是 ``ahew / eew``，即
**当量重比**；而 ``formulation_r_value`` 是**化学计量比 r**。两者语义不同、
数值不同（DGEBA/DDS 100:33 时 cp_r_value≈0.365，正确 r≈0.905），
互相替代会造成系统性预测偏差。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

__all__ = [
    "FormulationComponent",
    "derive_formulation_features",
    "derive_r_value",
    "parse_component_smiles",
]


class FormulationInputError(ValueError):
    """配方输入非法（SMILES 无效、phr 缺失等）—— 必须显式报错，不得静默。"""


#: 组分角色的中文展示名（仅用于报错信息，不参与逻辑判断）
_ROLE_LABEL = {"resin": "树脂", "hardener": "固化剂"}


def parse_component_smiles(smiles: str) -> list[str]:
    """把多组分 SMILES 拆成组件列表。

    ``.`` 是 SMILES 的多组分分隔符（如 ``"CC.OO"`` = 乙醇 + 水）。
    返回去空白后的非空组件；输入为空返回空列表。
    """
    text = str(smiles or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.split(".") if part.strip()]


def _engine():
    from core.epoxy_mechanism_features import EpoxyMechanismEngine

    return EpoxyMechanismEngine()


def _validate_smiles(engine, smiles: str, *, role: str, index: int) -> dict[str, Any]:
    """校验单个 SMILES 并取回性质；无效即抛错（不静默跳过）。

    ``role`` 必须是 ``"resin"`` 或 ``"hardener"``（机器可读值）；
    中文展示名由 ``_ROLE_LABEL`` 映射，不参与逻辑判断。

    ⚠️ 必须先自己做 RDKit 语法校验：实测 ``EpoxyMechanismEngine`` 对非法
    SMILES（如 ``"this_is_not_a_smiles(((("``）**不抛错**，而是返回一套
    伪造的兜底值（mw=360 / ew=180 / functionality=2）。若直接信任引擎，
    非法输入会静默产出错误预测。
    """
    if role not in {"resin", "hardener"}:
        raise ValueError(f"未知组分角色：{role!r}")
    label = _ROLE_LABEL[role]
    if not _is_parseable_smiles(smiles):
        raise FormulationInputError(
            f"{label}第 {index} 个组分的 SMILES 无法解析：{smiles!r}（请检查括号配对与原子价态）"
        )
    try:
        props = engine.calc_single_molecule_properties(
            smiles, is_resin=(role == "resin")
        )
    except Exception as exc:
        raise FormulationInputError(
            f"{label}第 {index} 个组分的 SMILES 无法解析：{smiles!r}（{exc}）"
        ) from exc
    if not isinstance(props, Mapping):
        raise FormulationInputError(
            f"{label}第 {index} 个组分的 SMILES 无法解析：{smiles!r}"
        )
    return dict(props)


def _is_parseable_smiles(smiles: str) -> bool:
    """用 RDKit 校验 SMILES 语法；不可解析或为空则 False。"""
    text = str(smiles or "").strip()
    if not text:
        return False
    try:
        from rdkit import Chem
    except Exception:  # pragma: no cover - RDKit 是项目必需依赖
        return True  # 无法校验时不阻断（由引擎自行报错）
    try:
        return Chem.MolFromSmiles(text) is not None
    except Exception:
        return False


class FormulationComponent:
    """配方中的一个组分（已校验）。"""

    __slots__ = ("role", "smiles", "phr", "props", "index")

    def __init__(self, *, role: str, smiles: str, phr: float, props: Mapping[str, Any], index: int):
        self.role = role
        self.smiles = smiles
        self.phr = float(phr)
        self.props = dict(props)
        self.index = index

    @property
    def equivalent_weight(self) -> float | None:
        """当量重：树脂为 EEW，固化剂为 AHEW。"""
        value = self.props.get("ew")
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    @property
    def functionality(self) -> float | None:
        value = self.props.get("functionality")
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None


def _weighted_harmonic_mean(
    components: Sequence[FormulationComponent],
) -> float | None:
    """质量加权调和平均当量重 = (Σphr) / (Σ phr_i / ew_i)。

    等价于 phr 加权，与 core/component_physics.py 的 ``_eff_ew`` 同口径
    （实测 DGEBA 单组分 = 170.2095，与 cp_eew 一致）。

    当 phr 缺失（全为 0）时：单组分退化为该组分自身的当量重（EEW/AHEW 是
    分子属性，与 phr 无关）；多组分无法确定加权 → 返回 None（不猜）。
    """
    usable = [c for c in components if c.equivalent_weight is not None]
    if not usable:
        return None
    total_phr = sum(c.phr for c in usable if c.phr > 0)
    if total_phr <= 0:
        if len(usable) == 1:
            return usable[0].equivalent_weight
        return None
    denominator = sum(
        c.phr / c.equivalent_weight for c in usable if c.phr > 0 and c.equivalent_weight
    )
    if denominator <= 0:
        return None
    return total_phr / denominator


def derive_r_value(
    *,
    resin_phr: float,
    hardener_phr: float,
    resin_eew: float | None,
    hardener_ahew: float | None,
) -> float | None:
    """化学计量比 r = (固化剂phr / AHEW) / (树脂phr / EEW)。

    返回 None 表示输入不足（任一参数缺失/非正）——**不得**猜 0 或均值。
    """
    try:
        resin_phr = float(resin_phr)
        hardener_phr = float(hardener_phr)
        resin_eew = float(resin_eew) if resin_eew is not None else None
        hardener_ahew = float(hardener_ahew) if hardener_ahew is not None else None
    except (TypeError, ValueError):
        return None
    if resin_eew is None or hardener_ahew is None:
        return None
    if resin_eew <= 0 or hardener_ahew <= 0 or resin_phr <= 0:
        return None
    return (hardener_phr / hardener_ahew) / (resin_phr / resin_eew)


def _derived(feature: str, value: Any, detail: str) -> dict[str, Any]:
    return {"value": value, "origin": "derived", "detail": detail}


def _sum_functionality(
    components: Sequence[FormulationComponent], *, per_component: bool
) -> float | None:
    """官能团总数：per_component=True 时按组分数累加官能度，否则按 phr 加权。"""
    total = 0.0
    seen = False
    for component in components:
        functionality = component.functionality
        if functionality is None:
            continue
        seen = True
        total += functionality
    return total if seen else None


def derive_formulation_features(
    *,
    resin_smiles: str,
    hardener_smiles: str,
    resin_phr: float,
    hardener_phr: float,
    extra_phr: Mapping[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """从配方推出可自动计算的配方特征。

    参数
    ----
    resin_smiles / hardener_smiles : 可多组分，``.`` 分隔
    resin_phr / hardener_phr : 总 phr（多组分时按质量比例分摊到各组分）
    extra_phr : 其他粘料组分（活性稀释剂/增韧剂等）的 phr，用于
        ``formulation_epoxy_binder_total_phr``

    返回
    ----
    ``{feature_name: {"value": v, "origin": "derived", "detail": str}}``

    抛出
    ----
    FormulationInputError : SMILES 非法（必须显式报错，不得静默产出错误值）
    """
    engine = _engine()

    resin_parts = parse_component_smiles(resin_smiles)
    hardener_parts = parse_component_smiles(hardener_smiles)
    if not resin_parts:
        raise FormulationInputError("缺少树脂 SMILES（配方必填）。")
    if not hardener_parts:
        raise FormulationInputError("缺少固化剂 SMILES（配方必填）。")

    # 多组分时按等分 phr 处理（用户只给总 phr；等分是与全表 resin_100_basis 一致的口径）
    resin_share = float(resin_phr) / len(resin_parts)
    hardener_share = float(hardener_phr) / len(hardener_parts)

    resins = [
        FormulationComponent(
            role="resin",
            smiles=part,
            phr=resin_share,
            props=_validate_smiles(engine, part, role="resin", index=i + 1),
            index=i + 1,
        )
        for i, part in enumerate(resin_parts)
    ]
    hardeners = [
        FormulationComponent(
            role="hardener",
            smiles=part,
            phr=hardener_share,
            props=_validate_smiles(engine, part, role="hardener", index=i + 1),
            index=i + 1,
        )
        for i, part in enumerate(hardener_parts)
    ]

    resin_eew = _weighted_harmonic_mean(resins)
    hardener_ahew = _weighted_harmonic_mean(hardeners)
    r_value = derive_r_value(
        resin_phr=resin_phr,
        hardener_phr=hardener_phr,
        resin_eew=resin_eew,
        hardener_ahew=hardener_ahew,
    )

    out: dict[str, dict[str, Any]] = {}

    if resin_eew is not None:
        out["formulation_resin_total_eew_g_eq"] = _derived(
            "formulation_resin_total_eew_g_eq",
            round(resin_eew, 4),
            f"树脂质量加权调和平均 EEW（{len(resins)} 个组分）",
        )
    if hardener_ahew is not None:
        out["formulation_hardener_total_ahew_g_eq"] = _derived(
            "formulation_hardener_total_ahew_g_eq",
            round(hardener_ahew, 4),
            f"固化剂质量加权调和平均 AHEW（{len(hardeners)} 个组分）",
        )
    if r_value is not None:
        detail = "r = (固化剂phr/AHEW)/(树脂phr/EEW)"
        out["formulation_r_value"] = _derived("formulation_r_value", round(r_value, 6), detail)
        # 全表实测两列 100% 相同（n=3518），故同源
        out["formulation_resin_hardener_equivalent_ratio"] = _derived(
            "formulation_resin_hardener_equivalent_ratio", round(r_value, 6), detail
        )

    # ---- 计量汇总 ----
    out["resin_total_phr"] = _derived("resin_total_phr", float(resin_phr), "树脂 phr 求和")
    out["curing_agent_total_phr"] = _derived(
        "curing_agent_total_phr", float(hardener_phr), "固化剂 phr 求和"
    )
    extras = {str(k): float(v) for k, v in (extra_phr or {}).items()}
    binder = float(resin_phr) + float(hardener_phr) + sum(extras.values())
    out["formulation_epoxy_binder_total_phr"] = _derived(
        "formulation_epoxy_binder_total_phr", binder, "树脂 + 固化剂 + 其他粘料 phr"
    )

    # ---- 组分计数 ----
    out["resin_component_count"] = _derived(
        "resin_component_count", len(resins), "树脂组分数"
    )
    out["curing_agent_component_count"] = _derived(
        "curing_agent_component_count", len(hardeners), "固化剂组分数"
    )
    for name, key in (
        ("accelerator_component_count", "accelerator"),
        ("catalyst_component_count", "catalyst"),
        ("initiator_component_count", "initiator"),
        ("other_component_count", "other"),
        ("reactive_diluent_component_count", "reactive_diluent"),
        ("reactive_toughener_component_count", "reactive_toughener"),
        ("small_additive_component_count", "small_additive"),
    ):
        out[name] = _derived(name, 0, f"未提供 {key} 组分（计数 0）")

    # ---- 官能团汇总 ----
    resin_epoxy = _sum_functionality(resins, per_component=True)
    if resin_epoxy is not None:
        out["resin_epoxy_group_total"] = _derived(
            "resin_epoxy_group_total", resin_epoxy, "各树脂组分环氧基数求和"
        )
        out["resin_equivalent_group_total"] = _derived(
            "resin_equivalent_group_total", resin_epoxy, "树脂官能团总数"
        )
    hardener_h = _sum_functionality(hardeners, per_component=True)
    if hardener_h is not None:
        out["curing_agent_active_hydrogen_total"] = _derived(
            "curing_agent_active_hydrogen_total", hardener_h, "各固化剂组分活泼氢数求和"
        )
        out["curing_agent_equivalent_group_total"] = _derived(
            "curing_agent_equivalent_group_total", hardener_h, "固化剂官能团总数"
        )

    # ---- 其他 ----
    out["initiator_present"] = _derived("initiator_present", False, "未提供引发剂组分")
    out["reactive_toughener_total_phr"] = _derived(
        "reactive_toughener_total_phr", extras.get("reactive_toughener", 0.0), "增韧剂 phr"
    )
    out["accelerator_total_phr"] = _derived(
        "accelerator_total_phr", extras.get("accelerator", 0.0), "促进剂 phr"
    )
    return out
