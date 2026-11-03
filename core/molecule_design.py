"""Domain types, precursor catalogs, reaction templates, scaffold fission,
and combinatorial assembly for virtual molecule design in CFRP composites.

This module provides:
1. RingScaffold, LinkerBridge, RGroupSubstituent building blocks (24 rings, 20 linkers, 12 R-groups),
   all gated by user selection during scaffold fission.
2. Scaffold fission and combinatorial intermediate generator (diphenols, diamines,
   polyacids, star/tetragonal spatial cores, asymmetric hybrids).
3. 15 synthetic Reaction SMARTS templates (Epoxy incl. aliphatic OH/NH2, Amine/Mannich,
   Anhydride, BMI, CE, BOZ, Propargyl, Isocyanurate/TGIC, phenolic hardener retain, etc.).
   All templates use local adjacency patterns so spectator substituents survive;
   reactions that do not match a precursor yield no product (no fall-through fakes).
4. Strict quality gates, RDKit sanitization, per-class stoichiometry (EEW, AHEW with
   1 epoxy : 1 anhydride-group convention) and deterministic output ordering.
5. Seamless handshake with downstream formula-level high-throughput virtual screening (HTVS).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any

import pandas as pd

from .smiles_utils import normalize_chemical_string

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdChemReactions, rdMolDescriptors
    RDKIT_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    Descriptors = None
    rdChemReactions = None
    rdMolDescriptors = None
    RDKIT_AVAILABLE = False


ALLOWED_ELEMENTS = {1, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}

# 环状酸酐通用 SMARTS：兼容 RDKit 的芳构化感知（PMDA/BTDA 等稠环芳酐的羰基碳
# 会被 RDKit 感知为芳香原子，传统 "C(=O)OC(=O)" 脂肪碳模式完全匹配不上）。
# 计数时以唯一中心氧原子为准，避免对称环的双向匹配重复计数。
ANHYDRIDE_RING_SMARTS = "[o,OX2]1~[#6](=[OX1])~[#6]~[#6]~[#6](=[OX1])~1"


def count_anhydride_groups(mol) -> int:
    """Count cyclic anhydride groups via unique central oxygen atoms (aromatization-safe)."""
    if mol is None or not RDKIT_AVAILABLE:
        return 0
    patt = Chem.MolFromSmarts(ANHYDRIDE_RING_SMARTS)
    if patt is None:
        return 0
    try:
        return len({m[0] for m in mol.GetSubstructMatches(patt)})
    except Exception:
        return 0


# ==============================================================================
# 1. 四层化学积木库数据结构 (Building Block Definitions)
# ==============================================================================

@dataclass
class RingScaffold:
    """Rigid/semi-rigid ring scaffold representing aromatic, fused, cardo, hetero, or alicyclic rings."""
    ring_id: str
    name: str
    category: str  # "单环/苯系" | "稠环芳烃" | "卡基/大体积三维" | "含氮/硫杂环" | "脂环/笼状"
    smiles_core: str  # e.g. "c1ccccc1" with attachment handles or base scaffold
    description: str
    valency_sites: list[str] = field(default_factory=lambda: ["1,4", "1,3"])


@dataclass
class LinkerBridge:
    """Main-chain bridge connecting ring scaffolds."""
    linker_id: str
    name: str
    category: str  # "烃基/柔性" | "含氟/低介电" | "极性/耐热" | "杂原子/阻燃" | "星型/多官能中心"
    smiles_pattern: str  # e.g. "C(C)(C)", "S(=O)(=O)", "C(C(F)(F)F)(C(F)(F)F)"
    valency: int = 2  # 2 for linear, 3 for trimeric star, 4 for tetrameric
    description: str = ""


@dataclass
class RGroupSubstituent:
    """Side-chain substituent for tuning melting point, viscosity, polarity, or dielectric properties."""
    group_id: str
    name: str
    category: str  # "烷基" | "卤素/含氟" | "烷氧基" | "反应性/极性"
    smiles: str  # e.g. "C", "C(F)(F)F", "F", "OC"
    description: str = ""


# ------------------------------------------------------------------------------
# 24 类全量骨架环体系 (Ring Scaffolds)
# ------------------------------------------------------------------------------
RING_SCAFFOLDS: list[RingScaffold] = [
    # 单环与苯系
    RingScaffold("benzene", "苯环", "单环/苯系", "c1ccccc1", "最基础芳香刚性单元", ["1,4", "1,3", "1,2"]),
    RingScaffold("toluene", "甲苯基", "单环/苯系", "Cc1ccccc1", "甲基不对称取代，降低结晶度与熔点", ["1,4", "2,4"]),
    RingScaffold("xylene", "二甲苯基", "单环/苯系", "Cc1cccc(C)c1", "二甲基位阻结构，耐水解与低吸水率", ["2,4", "2,5"]),
    RingScaffold("mesitylene", "均三甲苯基", "单环/苯系", "Cc1cc(C)cc(C)c1", "三对称甲基高位阻耐热骨架", ["1,3,5"]),
    # 稠环芳烃与联苯系
    RingScaffold("naphthalene_14", "1,4-萘环", "稠环芳烃", "c1ccc2ccccc2c1", "刚性平面稠环，提升耐温与耐湿热模量", ["1,4"]),
    RingScaffold("naphthalene_15", "1,5-萘环", "稠环芳烃", "c1ccc2c(cccc2c1)", "对称1,5稠环，致密芳香堆积", ["1,5"]),
    RingScaffold("naphthalene_26", "2,6-萘环", "稠环芳烃", "c1cc2cc(ccc2cc1)", "线性延伸高取向液晶元骨架", ["2,6"]),
    RingScaffold("naphthalene_27", "2,7-萘环", "稠环芳烃", "c1cc2ccc(cc2cc1)", "折线型耐热低粘度稠环", ["2,7"]),
    RingScaffold("anthracene", "9,10-蒽环", "稠环芳烃", "c1ccc2cc3ccccc3cc2c1", "大共轭刚性平面，超高模量与耐热性", ["9,10"]),
    RingScaffold("phenanthrene", "菲环", "稠环芳烃", "c1ccc2c(c1)ccc1ccccc21", "弯曲稠环芳烃，兼具刚性与空间溶解性", ["2,7", "3,6"]),
    RingScaffold("pyrene", "芘环", "稠环芳烃", "c1cc2ccc3cccc4ccc(c1)c2c34", "四稠环大共轭，极高耐热与热力学稳定性", ["1,6", "2,7"]),
    RingScaffold("biphenyl_44", "4,4'-联苯", "稠环芳烃", "c1ccc(-c2ccccc2)cc1", "联苯液晶元刚性结构，高模量与高断裂韧性", ["4,4'"]),
    RingScaffold("biphenyl_33", "3,3'-联苯", "稠环芳烃", "c1cccc(-c2ccccc2)c1", "间位联苯扭转构象，宽加工窗口与高韧性", ["3,3'"]),
    RingScaffold("terphenyl", "对三联苯", "稠环芳烃", "c1ccc(-c2ccc(-c3ccccc3)cc2)cc1", "超长芳香液晶刚棒骨架", ["4,4''"]),
    # 卡基 (Cardo) 与大体积三维刚性环
    RingScaffold("fluorene", "芴基 (Fluorene)", "卡基/大体积三维", "c1ccc2c(c1)Cc1ccccc21", "五元环并双苯环，兼备刚性与高溶解性", ["9,9", "2,7"]),
    RingScaffold("cardo_bpf", "卡基芴 (Cardo Fluorene)", "卡基/大体积三维", "c1ccc2c(c1)C(c1ccccc1)(c1ccccc1)c1ccccc21", "大体积卡牌状立体基团，超高 Tg 与超低 CTE", ["9,9-双苯"]),
    RingScaffold("spirobiindane", "螺双二氢茚 (SBP)", "卡基/大体积三维", "CC1(C)Cc2cccc3c2C12CCCc1cccc(c12)C3(C)C", "正交扭转三维刚性自微孔骨架", ["螺环双位"]),
    RingScaffold("xanthene", "呫吨/氧杂蒽 (Xanthene)", "卡基/大体积三维", "c1ccc2Oc3ccccc3Cc2c1", "含氧杂三环刚性耐热骨架", ["3,6", "9,9"]),
    # 含氮/硫杂环
    RingScaffold("s_triazine", "对称三嗪 (s-Triazine)", "含氮/硫杂环", "c1ncnc(n1)", "高对称含氮杂环，高交联度、超高耐温与阻燃性", ["2,4,6"]),
    RingScaffold("pyridine", "吡啶 (Pyridine)", "含氮/硫杂环", "c1ccncc1", "极性含氮芳杂环，自催化交联活性与耐热性", ["2,6", "3,5"]),
    RingScaffold("carbazole", "咔唑 (Carbazole)", "含氮/硫杂环", "c1ccc2c(c1)[nH]c1ccccc21", "含氮大共轭杂环，优异的耐温与耐光老化性能", ["3,6", "N-9"]),
    RingScaffold("quinoxaline", "喹喔啉 (Quinoxaline)", "含氮/硫杂环", "c1ccc2nccnc2c1", "双氮稠环，极高玻璃化转变温度与耐辐射性", ["2,3", "6,7"]),
    # 脂环族与笼状烃
    RingScaffold("cyclohexane", "环己烷", "脂环/笼状", "C1CCCCC1", "完全氢化脂环，优良透光率、耐候性与低介电", ["1,4", "1,3"]),
    RingScaffold("adamantane", "金刚烷 (Adamantane)", "脂环/笼状", "C1C2CC3CC1CC(C2)C3", "高对称三维笼状金刚石亚单元，超低介电与高耐热", ["1,3", "1,3,5,7"]),
]


# ------------------------------------------------------------------------------
# 20 类全量主链连接桥联库 (Linker Bridges)
# ------------------------------------------------------------------------------
LINKER_BRIDGES: list[LinkerBridge] = [
    # 烃基与柔性桥
    LinkerBridge("direct_bond", "直接单键 (-)", "烃基/柔性", "", 2, "芳环直接共轭连接"),
    LinkerBridge("methylene", "亚甲基 (-CH2-)", "烃基/柔性", "C", 2, "标准双酚F型柔性低粘度连接桥"),
    LinkerBridge("isopropylidene", "异亚丙基 (-C(CH3)2-)", "烃基/柔性", "C(C)(C)", 2, "标准双酚A型综合性能桥"),
    LinkerBridge("diethyl_methylene", "二乙基亚甲基 (-C(Et)2-)", "烃基/柔性", "C(CC)(CC)", 2, "大体积烷基侧链，耐冲击与低熔点"),
    LinkerBridge("cyclohexylidene", "环己基叉 (-C6H10-)", "烃基/柔性", "C1(CCCCC1)", 2, "双酚Z型脂环刚性桥，高耐温与高透明度"),
    LinkerBridge("dcpd_bridge", "双环戊二烯烃基桥 (-DCPD-)", "烃基/柔性", "C1CC2CC1C1CCC21", 2, "大体积疏水脂环，极低吸水率与优良韧性"),
    # 含氟与低介电桥
    LinkerBridge("hexafluoroisopropylidene", "六氟异亚丙基 (-C(CF3)2-)", "含氟/低介电", "C(C(F)(F)F)(C(F)(F)F)", 2, "双酚AF型全氟异丙基桥，超低 Dk/Df、低吸水率"),
    LinkerBridge("perfluoro_ethylene", "全氟亚乙基 (-CF2CF2-)", "含氟/低介电", "C(F)(F)C(F)(F)", 2, "全氟柔性耐候链段"),
    # 耐热、极性与耐蠕变桥
    LinkerBridge("sulfone", "砜基 (-SO2-)", "极性/耐热", "S(=O)(=O)", 2, "双酚S型强极性高耐热桥，抗蠕变"),
    LinkerBridge("sulfoxide", "亚砜基 (-SO-)", "极性/耐热", "S(=O)", 2, "极性连接桥"),
    LinkerBridge("carbonyl", "羰基/酮基 (-CO-)", "极性/耐热", "C(=O)", 2, "二苯酮型刚性耐热桥"),
    LinkerBridge("ester", "酯键 (-COO-)", "极性/耐热", "C(=O)O", 2, "易加工低粘度酯基桥"),
    LinkerBridge("amide", "酰胺键 (-CONH-)", "极性/耐热", "C(=O)N", 2, "强氢键网络高模量桥"),
    # 杂原子与阻燃/增韧桥
    LinkerBridge("ether", "醚键 (-O-)", "杂原子/阻燃", "O", 2, "柔性耐热链段，降低体系熔点与粘度"),
    LinkerBridge("thioether", "硫醚键 (-S-)", "杂原子/阻燃", "S", 2, "高折射率与高韧性杂原子桥"),
    LinkerBridge("phenyl_phosphonate", "苯基磷氧桥 (-P(=O)(Ph)-)", "杂原子/阻燃", "P(=O)(c1ccccc1)", 2, "本征阻燃磷杂菲/磷酸酯耐热骨架"),
    LinkerBridge("siloxane", "四甲基二硅氧烷桥", "杂原子/阻燃", "[Si](C)(C)O[Si](C)(C)", 2, "有机硅耐高低温、低应力增韧桥"),
    # 多官能/星型与空间核心
    LinkerBridge("methine_star", "三官能次甲基核 (>CH-)", "星型/多官能中心", "C", 3, "三(羟苯基)甲烷型星型拓扑核心"),
    LinkerBridge("ethane_tetra", "四官能乙烷四基核 (>CH-CH<)", "星型/多官能中心", "CC", 4, "四(羟苯基)乙烷四维交联核心"),
    LinkerBridge("quaternary_carbon", "季碳四面体核 (>C<)", "星型/多官能中心", "C", 4, "季戊四醇/四苯甲烷空间超高交联核心"),
]


# ------------------------------------------------------------------------------
# 12 类侧链修饰 R 基团 (Substituents)
# ------------------------------------------------------------------------------
R_GROUPS: list[RGroupSubstituent] = [
    RGroupSubstituent("H", "氢原子 (-H)", "基础", "", "无侧链修饰"),
    RGroupSubstituent("methyl", "甲基 (-CH3)", "烷基", "C", "位阻效应，提升耐水解性"),
    RGroupSubstituent("t_butyl", "叔丁基 (-tBu)", "烷基", "C(C)(C)C", "大体积位阻，降低结晶度与熔点"),
    RGroupSubstituent("fluoro", "氟原子 (-F)", "卤素/含氟", "F", "强吸电子性，降低极化率与介电常数"),
    RGroupSubstituent("trifluoromethyl", "三氟甲基 (-CF3)", "卤素/含氟", "C(F)(F)F", "超低介电常数与高热氧化稳定性"),
    RGroupSubstituent("chloro", "氯原子 (-Cl)", "卤素/含氟", "Cl", "本征自熄阻燃特性"),
    RGroupSubstituent("bromo", "溴原子 (-Br)", "卤素/含氟", "Br", "高效阻燃基团"),
    RGroupSubstituent("methoxy", "甲氧基 (-OCH3)", "烷氧基", "OC", "给电子基团，提升电荷分布均匀性"),
    RGroupSubstituent("trifluoromethoxy", "三氟甲氧基 (-OCF3)", "烷氧基", "OC(F)(F)F", "极低表面能与超低介电损耗"),
    RGroupSubstituent("vinyl", "乙烯基 (-CH=CH2)", "反应性/极性", "C=C", "双键侧基，提供自由基热后交联点"),
    RGroupSubstituent("allyl", "烯丙基 (-CH2-CH=CH2)", "反应性/极性", "CC=C", "烯丙基增韧并参与加成共聚"),
    RGroupSubstituent("cyano", "氰基 (-CN)", "反应性/极性", "C#N", "强偶极基团，提升热变形温度与界面粘接力"),
]


# ==============================================================================
# 2. 真实有机反应模板库 (12 大多通道 Reaction SMARTS)
# ==============================================================================

@dataclass
class SyntheticReaction:
    """Rigorous chemical reaction mapping from precursor to target monomer."""
    reaction_id: str
    name: str
    target_role: str  # "resin" | "hardener" | "both"
    reaction_smarts: str
    description: str
    expected_warhead: str
    chemical_system: str  # "epoxy" | "amine" | "anhydride" | "bmi" | "cyanate" | "benzoxazine" | "propargyl" | "phenol"
    default_enabled: bool = True


SYNTHETIC_REACTION_TEMPLATES: list[SyntheticReaction] = [
    SyntheticReaction(
        reaction_id="glycidyl_etherification",
        name="羟基缩水甘油醚化 (R01, 酚/脂肪醇通用)",
        target_role="resin",
        reaction_smarts="[#6;!$(C=O):1]-[OX2H:2]>>[#6:1]-OCC1CO1",
        description="多元酚/脂肪多元醇 + 环氧氯丙烷生成缩水甘油醚环氧树脂 (DGEBA/氢化双酚A/脂环多元醇醚型)",
        expected_warhead="环氧缩水甘油醚",
        chemical_system="epoxy",
        default_enabled=True,
    ),
    SyntheticReaction(
        reaction_id="glycidyl_etherification_non_carboxyl",
        name="羟基缩水甘油醚化·排除羧酸 (R01c)",
        target_role="resin",
        reaction_smarts="[#6;!$(C=O):1]-[OX2H:2]>>[#6:1]-OCC1CO1",
        description="酚/脂肪醇缩水甘油醚化，显式排除羧酸（与 R03 分工，供 P1 路线链引用）",
        expected_warhead="环氧缩水甘油醚",
        chemical_system="epoxy",
        default_enabled=True,
    ),
    SyntheticReaction(
        reaction_id="glycidyl_etherification_phenol",
        name="酚羟基缩水甘油醚化 (R01a)",
        target_role="resin",
        reaction_smarts="[c:1]-[OX2H:2]>>[c:1]-OCC1CO1",
        description="仅酚羟基 + 环氧氯丙烷生成缩水甘油醚",
        expected_warhead="环氧缩水甘油醚",
        chemical_system="epoxy",
        default_enabled=True,
    ),
    SyntheticReaction(
        reaction_id="glycidyl_etherification_aliphatic",
        name="脂肪醇缩水甘油醚化 (R01b)",
        target_role="resin",
        reaction_smarts="[CX4:1]-[OX2H:2]>>[CX4:1]-OCC1CO1",
        description="仅脂肪醇 + 环氧氯丙烷生成缩水甘油醚",
        expected_warhead="环氧缩水甘油醚",
        chemical_system="epoxy",
        default_enabled=True,
    ),
    SyntheticReaction(
        reaction_id="glycidyl_amination",
        name="芳香胺缩水甘油胺化 (R02)",
        target_role="resin",
        reaction_smarts="[c:1][NX3;H2:2]>>[c:1]N(CC2CO2)CC3CO3",
        description="芳香伯胺 + 环氧氯丙烷生成多官能缩水甘油胺型环氧树脂 (TGDDM/AFG-90 系列)",
        expected_warhead="缩水甘油胺",
        chemical_system="epoxy",
        default_enabled=True,
    ),
    SyntheticReaction(
        reaction_id="glycidyl_amination_aliphatic",
        name="脂肪胺缩水甘油胺化 (R02b)",
        target_role="resin",
        reaction_smarts="[CX4:1][NX3;H2:2]>>[CX4:1]N(CC2CO2)CC3CO3",
        description="脂肪/脂环伯胺 + 环氧氯丙烷生成低粘度脂肪族缩水甘油胺树脂",
        expected_warhead="脂肪缩水甘油胺",
        chemical_system="epoxy",
        default_enabled=True,
    ),
    SyntheticReaction(
        reaction_id="glycidyl_esterification",
        name="羧酸缩水甘油酯化 (R03)",
        target_role="resin",
        reaction_smarts="[C:1](=O)[OX2H:2]>>[C:1](=O)OCC2CO2",
        description="多元羧酸 + 环氧氯丙烷生成缩水甘油酯低粘度树脂",
        expected_warhead="缩水甘油酯",
        chemical_system="epoxy",
        default_enabled=True,
    ),
    SyntheticReaction(
        reaction_id="olefin_epoxidation",
        name="双键环氧化 (R04, 脂环环氧)",
        target_role="resin",
        reaction_smarts=(
            "[!$([CX3]=[OX1]);!$([CX4,CX3][CX3]=[OX1]);!$([CX4,CX3][CX2]#[NX1]):1]"
            "=[!$([CX3]=[OX1]);!$([CX4,CX3][CX3]=[OX1]);!$([CX4,CX3][CX2]#[NX1]):2]"
            ">>[C:1]1O[C:2]1"
        ),
        description="过氧酸对脂肪族双键（环己烯/乙烯基/烯丙基侧链）环氧化生成耐候脂环族环氧（局部模板，保留环上取代基）",
        expected_warhead="脂环环氧",
        chemical_system="epoxy",
        default_enabled=True,
    ),
    SyntheticReaction(
        reaction_id="mannich_polyamine",
        name="酚醛胺曼尼希反应 (R05, 改性胺固化剂)",
        target_role="hardener",
        reaction_smarts="[OX2H:1][c:2][cH:3]>>[OX2H:1][c:2][c:3](CNCCN)",
        description="多元酚邻位 + 甲醛 + 乙二胺曼尼希缩合生成酚醛胺快速低温固化剂（邻位氨甲基化）",
        expected_warhead="酚醛改性胺活性氢",
        chemical_system="amine",
        default_enabled=True,
    ),
    SyntheticReaction(
        reaction_id="amine_alkylation",
        name="芳香胺烷基化/增韧改性 (R06)",
        target_role="hardener",
        reaction_smarts="[c:1][NX3;H2:2]>>[c:1]NCC",
        description="调控多元胺反应活性与交联网络自由体积",
        expected_warhead="仲胺活性氢",
        chemical_system="amine",
        default_enabled=True,
    ),
    SyntheticReaction(
        reaction_id="anhydride_cyclization_aromatic",
        name="芳香邻二羧酸脱水酸酐化 (R07a)",
        target_role="hardener",
        reaction_smarts="[C:1](=O)([OH])[c:4][c:3][C:2](=O)[OH]>>[C:1]1(=O)O[C:2](=O)[c:3][c:4]1",
        description="芳香邻位二羧酸分子内脱水生成环状酸酐固化剂（邻苯二甲酸酐/PMDA/BTDA 型）",
        expected_warhead="酸酐基",
        chemical_system="anhydride",
        default_enabled=True,
    ),
    SyntheticReaction(
        reaction_id="anhydride_cyclization_aliphatic",
        name="脂肪1,2-二羧酸脱水酸酐化 (R07b)",
        target_role="hardener",
        reaction_smarts="[C:1](=O)([OH])[C;X4:4][C;X4:3][C:2](=O)[OH]>>[C:1]1(=O)O[C:2](=O)[C:3][C:4]1",
        description="相邻 sp3 碳上的 1,2-二羧酸分子内脱水生成五元环酸酐（琥珀酸酐/四氢苯酐/六氢苯酐型）",
        expected_warhead="酸酐基",
        chemical_system="anhydride",
        default_enabled=True,
    ),
    SyntheticReaction(
        reaction_id="bismaleimide_synthesis",
        name="双马来酰亚胺化 (R08, BMI树脂)",
        target_role="resin",
        reaction_smarts="[c:1][NX3;H2:2]>>[c:1]N1C(=O)C=CC1=O",
        description="芳香二胺 + 顺酐生成双马来酰亚胺 (BMI) 超高耐温复合材料基体树脂",
        expected_warhead="马来酰亚胺基",
        chemical_system="bmi",
        default_enabled=False,
    ),
    SyntheticReaction(
        reaction_id="cyanate_esterification",
        name="酚羟基氰酸酯化 (R09, CE树脂)",
        target_role="resin",
        reaction_smarts="[c:1][OX2H:2]>>[c:1]OC#N",
        description="多元酚 + 卤化氰生成氰酸酯树脂（超低介电损耗 Dk/Df、高 Tg）",
        expected_warhead="氰酸酯 (-OCN)",
        chemical_system="cyanate",
        default_enabled=False,
    ),
    SyntheticReaction(
        reaction_id="benzoxazine_synthesis",
        name="苯并噁嗪环化 (R10, BOZ树脂)",
        target_role="resin",
        reaction_smarts="[OX2H:1][c:2][cH:3]>>[OX2:1]1[C:5][N:6](c2ccccc2)[C:7][c:3][c:2]1",
        description="多元酚邻位 + 苯胺 + 甲醛缩合生成苯并噁嗪单体（近零固化收缩率，N-苯基型）",
        expected_warhead="苯并噁嗪环",
        chemical_system="benzoxazine",
        default_enabled=False,
    ),
    SyntheticReaction(
        reaction_id="propargyl_etherification",
        name="酚羟基炔丙基醚化 (R11, 炔丙基树脂)",
        target_role="resin",
        reaction_smarts="[c:1][OX2H:2]>>[c:1]OCC#C",
        description="多元酚 + 炔丙基溴生成双炔丙基醚树脂（无挥发分自催化聚合、高耐热）",
        expected_warhead="炔丙基醚",
        chemical_system="propargyl",
        default_enabled=False,
    ),
    SyntheticReaction(
        reaction_id="native_hardener_retain",
        name="天然固化剂母核直接继承 (R12)",
        target_role="hardener",
        reaction_smarts="",
        description="保留芳香多胺、脂环胺、多元酸酐、双氰胺、咪唑、硫醇母核本身，并自动计算活性氢/酸酐当量",
        expected_warhead="活性胺氢/酸酐基",
        chemical_system="amine",
        default_enabled=True,
    ),
    SyntheticReaction(
        reaction_id="isocyanurate_glycidylation",
        name="异氰脲酸酯/酰亚胺 N-缩水甘油化 (R13, TGIC型)",
        target_role="resin",
        reaction_smarts="[nH:1][c:2](=[OX1])>>[n:1](CC1CO1)[c:2](=[OX1])",
        description="氰尿酸(异氰脲酸)/酰亚胺 N-H + 环氧氯丙烷生成三缩水甘油异氰脲酸酯 (TGIC) 耐候粉末涂料树脂",
        expected_warhead="异氰脲酸酯缩水甘油",
        chemical_system="epoxy",
        default_enabled=True,
    ),
    SyntheticReaction(
        reaction_id="phenolic_hardener_retain",
        name="多元酚/酚醛固化剂直接继承 (R14)",
        target_role="hardener",
        reaction_smarts="",
        description="多元酚与酚醛低聚物作为酚羟基型环氧固化剂直接入库，按酚羟基数计算当量",
        expected_warhead="酚羟基",
        chemical_system="phenol",
        default_enabled=True,
    ),
]


# ==============================================================================
# 2b. 路线感知合成路径 (Synthetic Routes)：将扁平反应模板组织为多步反应网络
# ==============================================================================

@dataclass
class RouteStep:
    """合成路线中的单个反应步骤。"""
    step_index: int
    reaction_id: str
    reaction_smarts: str
    intermediate_label: str = ""           # 中间体的可读标签（仅用于 UI 展示，非 SMILES）
    requires_reagent: str = ""             # "环氧氯丙烷" / "烯丙基溴" / "过氧酸" / "NaOH" ...
    hazard_flags: tuple[str, ...] = ()     # ("剧毒", "含氯废水", "强放热")


@dataclass
class SyntheticRoute:
    """从前驱体到目标单体的完整合成路线。

    steps 顺序执行：step[i] 的产物是 step[i+1] 的底物。
    中间体的真实 SMILES 由路径执行器在执行时产出（见 _expand_route_chain），
    不存放在本结构中；step.intermediate_label 仅为 UI 可读标签。
    """
    route_id: str
    name: str
    chemical_system: str                  # "epoxy" | "amine" | ...
    target_role: str                      # "resin" | "hardener"
    steps: list[RouteStep]
    atom_economy: float = 0.5             # 原子经济性 (0~1)
    green_score: float = 5.0              # 绿色化学评分 (0~10)
    default_enabled: bool = True

    @property
    def n_steps(self) -> int:
        return len(self.steps)


# 羟基环氧化四条真实工业路线（P1-P4），均默认启用
_P1_SMARTS = "[#6;!$(C=O):1]-[OX2H:2]>>[#6:1]-OCC1CO1"

ROUTE_CHAINS: list[SyntheticRoute] = [
    SyntheticRoute(
        route_id="P1_phenol_ech",
        name="P1 直接ECH醚化 (酚/醇→缩水甘油醚)",
        chemical_system="epoxy",
        target_role="resin",
        steps=[
            RouteStep(0, "glycidyl_etherification_non_carboxyl", _P1_SMARTS,
                      requires_reagent="环氧氯丙烷", hazard_flags=("剧毒", "含氯废水")),
        ],
        atom_economy=0.42, green_score=4.5,
    ),
    SyntheticRoute(
        route_id="P2_allyl_epoxidation",
        name="P2 烯丙基化-环氧化 (两步, 绿色无氯)",
        chemical_system="epoxy",
        target_role="resin",
        steps=[
            RouteStep(0, "allyl_etherification",
                      "[c:1][OX2H:2]>>[c:1]OCC=C",
                      requires_reagent="烯丙基溴", hazard_flags=("催泪",)),
            RouteStep(1, "olefin_epoxidation_selective",
                      "[!$([CX3]=[OX1]);!$([CX4,CX3][CX3]=[OX1]);!$([CX4,CX3][CX2]#[NX1]):1]"
                      "=[!$([CX3]=[OX1]);!$([CX4,CX3][CX3]=[OX1]);!$([CX4,CX3][CX2]#[NX1]):2]"
                      ">>[C:1]1O[C:2]1",
                      intermediate_label="O-烯丙基醚中间体",
                      requires_reagent="过氧酸", hazard_flags=("易爆", "强氧化")),
        ],
        atom_economy=0.55, green_score=6.0,
    ),
    SyntheticRoute(
        route_id="P3_chlorohydrin",
        name="P3 氯代醇法闭环 (两步, 机理清晰)",
        chemical_system="epoxy",
        target_role="resin",
        steps=[
            # 首步 SMARTS 刻意用 ;!$(C=O) 排除羧基碳，避免羧酸被本步消耗（与 R01c 选择性修正一致）。
            RouteStep(0, "chlorohydrin_ether",
                      "[#6;!$(C=O):1][OX2H:2]>>[#6:1]OCC(O)CCl",
                      requires_reagent="环氧氯丙烷", hazard_flags=("剧毒",)),
            RouteStep(1, "chlorohydrin_closure",
                      "[OX2:1][CH2:2][CH:3]([OX2H])[CH2:4][Cl:5]>>[OX2:1][CH2:2][CH:3]1[O][CH2:4]1",
                      intermediate_label="氯代醇中间体",
                      requires_reagent="NaOH", hazard_flags=("含盐废水",)),
        ],
        atom_economy=0.48, green_score=5.0,
    ),
    SyntheticRoute(
        route_id="P4_sharpless",
        name="P4 Sharpless环氧化 (消旋, 烯丙醇→环氧醇)",
        chemical_system="epoxy",
        target_role="resin",
        steps=[
            RouteStep(0, "sharpless_epoxidation",
                      "[CX4:1]([OX2H:2])[CX3:3]=[CX3:4]>>[CX4:1]([OX2H:2])[CX3:3]1[O][CX3:4]1",
                      intermediate_label="环氧醇",
                      requires_reagent="Ti(OiPr)4/TBHP", hazard_flags=("重金属", "强氧化")),
        ],
        atom_economy=0.60, green_score=5.5,
    ),
]


def hydroxyl_epoxidation_routes() -> list[SyntheticRoute]:
    """返回全部'羟基→环氧'转化路径（供 UI 与流水线引用）。"""
    return [r for r in ROUTE_CHAINS if r.chemical_system == "epoxy"]


# ==============================================================================
# 3. 经典工业前驱体母核库 (Precursor Catalog)
# ==============================================================================

@dataclass
class PrecursorCore:
    """Precursor chemical building block representing industrial or advanced monomers."""
    core_id: str
    name: str
    category: str
    role: str  # "resin" | "hardener" | "both"
    smiles: str
    description: str = ""
    default_selected: bool = True


PRECURSOR_CATALOG: list[PrecursorCore] = [
    # 树脂前驱体：多元酚与多元酸
    PrecursorCore("bpa", "双酚A (BPA)", "双酚系列", "resin", "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1", "最通用环氧树脂骨架，优良的综合力学与加工性", True),
    PrecursorCore("bpf", "双酚F (BPF)", "双酚系列", "resin", "Oc1ccc(Cc2ccc(O)cc2)cc1", "低粘度树脂母核，适合高流动性灌注与浸润", True),
    PrecursorCore("bps", "双酚S (BPS)", "双酚系列", "resin", "Oc1ccc(S(=O)(=O)c2ccc(O)cc2)cc1", "极性砜基骨架，优异的耐热性与抗蠕变性", True),
    PrecursorCore("bpaf", "六氟双酚A (BPAF)", "特种/低介电", "resin", "Oc1ccc(C(C(F)(F)F)(C(F)(F)F)c2ccc(O)cc2)cc1", "全氟异丙基桥，超低介电常数、低吸水率与阻燃性", True),
    PrecursorCore("tmbpa", "四甲基双酚A (TMBPA)", "双酚系列", "resin", "Cc1cc(C(C)(C)c2cc(C)c(O)c(C)c2)cc(C)c1O", "邻位甲基位阻，提升耐水解与耐热冲击性", True),
    PrecursorCore("bhpf", "芴基双酚 (BHPF / 双酚芴)", "特种/耐高温", "resin", "Oc1ccc(C2(c3ccccc3-c3ccccc32)c2ccc(O)cc2)cc1", "大体积卡牌状卡基芴环，超高 Tg、低 CTE", True),
    PrecursorCore("15_dhn", "1,5-萘二酚 (1,5-DHN)", "稠环芳香类", "resin", "Oc1cccc2c(O)cccc12", "致密萘环刚性平面，卓越的高温模量与耐湿热性", True),
    PrecursorCore("27_dhn", "2,7-萘二酚 (2,7-DHN)", "稠环芳香类", "resin", "Oc1ccc2cc(O)ccc2c1", "对称萘环双酚，耐热结晶性单体", True),
    PrecursorCore("resorcinol", "间苯二酚 (Resorcinol)", "稠环芳香类", "resin", "Oc1cccc(O)c1", "间位双官能活性母核，超低粘度、超高交联密度", True),
    PrecursorCore("biphenol", "4,4'-联苯二酚 (BP)", "稠环芳香类", "resin", "Oc1ccc(-c2ccc(O)cc2)cc1", "联苯液晶元刚性结构，优异的断裂韧性与高模量", True),
    PrecursorCore("thpm", "三(4-羟苯基)甲烷 (THPM)", "高官能度", "resin", "Oc1ccc(C(c2ccc(O)cc2)c2ccc(O)cc2)cc1", "三官能度星型多酚，构建致密 3D 网状交联", True),
    PrecursorCore("thpe", "四(4-羟苯基)乙烷 (THPE)", "高官能度", "resin", "Oc1ccc(C(c2ccc(O)cc2)C(c2ccc(O)cc2)c2ccc(O)cc2)cc1", "四官能度超高交联单体母核", True),
    PrecursorCore("isophthalic", "间苯二甲酸 (Isophthalic Acid)", "多元酸/酯类", "resin", "O=C(O)c1cccc(C(=O)O)c1", "缩水甘油酯树脂前驱体，极低粘度与高浸润性", True),
    PrecursorCore("dcpd_diphenol", "双环戊二烯双酚 (DCPD-Phenol)", "特种/低介电", "resin", "Oc1ccc(C2CC3CC2C2CCC32c2ccc(O)cc2)cc1", "大体积疏水烃基桥，极低吸水率与优良耐湿热性", True),
    # 固化剂前驱体：芳香多胺、脂环胺与酸酐
    PrecursorCore("ddm", "4,4'-二氨基二苯甲烷 (DDM)", "芳香多胺", "hardener", "Nc1ccc(Cc2ccc(N)cc2)cc1", "工业经典高性能耐热芳香胺固化剂母核", True),
    PrecursorCore("44_dds", "4,4'-二氨基二苯砜 (4,4'-DDS)", "芳香多胺", "hardener", "Nc1ccc(S(=O)(=O)c2ccc(N)cc2)cc1", "航空航天主承力 CFRP 标杆固化剂，超高耐温与湿热保持率", True),
    PrecursorCore("33_dds", "3,3'-二氨基二苯砜 (3,3'-DDS)", "芳香多胺", "hardener", "Nc1cccc(S(=O)(=O)c2cccc(N)c2)c1", "间位砜基二胺，相较 4,4'-DDS 拥有更佳韧性与操作窗口", True),
    PrecursorCore("oda", "4,4'-二氨基二苯醚 (ODA)", "芳香多胺", "hardener", "Nc1ccc(Oc2ccc(N)cc2)cc1", "柔性醚键连接芳香胺，兼顾耐热与韧性", True),
    PrecursorCore("m_pda", "间苯二胺 (m-PDA)", "芳香多胺", "hardener", "Nc1cccc(N)c1", "小分子高活性芳香胺，极高交联密度", True),
    PrecursorCore("p_pda", "对苯二胺 (p-PDA)", "芳香多胺", "hardener", "Nc1ccc(N)cc1", "高刚性共轭对位二胺", True),
    PrecursorCore("dab", "4,4'-二氨基联苯 (DAB)", "芳香多胺", "hardener", "Nc1ccc(-c2ccc(N)cc2)cc1", "刚性联苯二胺，提升基体模量与 Tg", True),
    PrecursorCore("fda", "9,9-双(4-氨基苯基)芴 (FDA / 芴二胺)", "特种/耐高温", "hardener", "Nc1ccc(C2(c3ccccc3-c3ccccc32)c2ccc(N)cc2)cc1", "卡基芴耐高温二胺，Tg > 220°C 标配单体", True),
    PrecursorCore("ipda", "异佛尔酮二胺 (IPDA)", "脂环族", "hardener", "CC1(C)CC(C)(CN)CC(N)C1", "不对称脂环胺，优良韧性与中温固化性", True),
    PrecursorCore("detda", "二乙基甲苯二胺 (DETDA)", "芳香多胺", "hardener", "CCc1cc(C)c(N)c(CC)c1N", "空间位阻芳香二胺，长适用期与高耐温性", True),
    PrecursorCore("mthpa", "甲基四氢苯酐 (MTHPA)", "多元酸酐", "hardener", "CC1=CCC2C(=O)OC(=O)C2C1", "低粘度长适用期酸酐，优异电绝缘性能", True),
    PrecursorCore("pmda", "均苯四甲酸二酐 (PMDA)", "多元酸酐", "hardener", "O=C1OC(=O)c2cc3C(=O)OC(=O)c3cc21", "四元芳香二酐，超高交联与耐温特性", True),
    # 单环二元酚与酚醛低聚物（线性酚醛环氧/固化剂母核）
    PrecursorCore("hydroquinone", "对苯二酚 (HQ)", "单环二酚", "resin", "Oc1ccc(O)cc1", "对位双官能母核，低粘度缩水甘油醚与链扩展剂", True),
    PrecursorCore("catechol", "邻苯二酚 (Catechol)", "单环二酚", "resin", "Oc1ccccc1O", "邻位双官能母核，可环碳酸酯/苯并噁嗪化衍生", True),
    PrecursorCore("novolac_trimer", "线性酚醛三核体 (Novolac n=1)", "酚醛低聚物", "resin", "Oc1ccccc1Cc1ccc(Cc2ccccc2O)cc1", "酚醛清漆三核低聚物，环氧酚醛(EOCN)与酚醛固化剂母核", True),
    PrecursorCore("novolac_tetramer", "线性酚醛四核体 (Novolac n=2)", "酚醛低聚物", "resin", "Oc1ccccc1Cc1ccc(Cc2ccc(Cc3ccccc3O)cc2)cc1", "酚醛清漆四核低聚物，高官能度环氧酚醛母核", True),
    # 多元羧酸与酸酐前驱体（缩水甘油酯树脂 + 酸酐固化剂双路线）
    PrecursorCore("phthalic_acid", "邻苯二甲酸 (PA)", "多元酸/酯类", "resin", "O=C(O)c1ccccc1C(=O)O", "邻位二酸，分子内脱水生成邻苯二甲酸酐，亦可缩水甘油酯化", True),
    PrecursorCore("trimellitic_acid", "偏苯三甲酸 (TMA)", "多元酸/酯类", "resin", "O=C(O)c1ccc(C(=O)O)c(C(=O)O)c1", "1,2,4-三酸，偏酐/三缩水甘油酯双路线母核", True),
    PrecursorCore("pyromellitic_acid", "均苯四甲酸 (PMA)", "多元酸/酯类", "resin", "O=C(O)c1cc(C(=O)O)c(C(=O)O)cc1C(=O)O", "1,2,4,5-四酸，双分子内脱水生成 PMDA，亦可四缩水甘油酯化", True),
    PrecursorCore("trimesic_acid", "均苯三甲酸 (TMA-135)", "多元酸/酯类", "resin", "O=C(O)c1cc(C(=O)O)cc(C(=O)O)c1", "1,3,5-对称三酸，三官能缩水甘油酯低粘度树脂母核", True),
    PrecursorCore("btda_acid", "二苯酮四甲酸 (BTDA前体)", "多元酸/酯类", "resin", "O=C(O)c1ccc(C(=O)c2ccc(C(=O)O)c(C(=O)O)c2)cc1C(=O)O", "二苯酮-3,3',4,4'-四甲酸，双脱水生成 BTDA 二酐耐温固化剂", True),
    PrecursorCore("thpa_acid", "四氢邻苯二甲酸 (THPA前体)", "多元酸/酯类", "resin", "O=C(O)C1CC=CCC1C(=O)O", "环己烯-1,2-二羧酸，脱水生成四氢苯酐，双键可进一步环氧化", True),
    PrecursorCore("hpa_acid", "六氢邻苯二甲酸 (HHPA前体)", "多元酸/酯类", "resin", "O=C(O)C1CCCCC1C(=O)O", "环己烷-1,2-二羧酸，脱水生成六氢苯酐耐候固化剂", True),
    # 杂环活性母核（TGIC/三聚氰胺路线）
    PrecursorCore("cyanuric_acid", "氰尿酸/异氰脲酸 (CYA)", "杂环活性母核", "resin", "O=C1NC(=O)NC(=O)N1", "酮式三嗪三酮，N-缩水甘油化生成 TGIC 耐候粉末涂料树脂", True),
    PrecursorCore("melamine", "三聚氰胺 (Melamine)", "杂环活性母核", "hardener", "Nc1nc(N)nc(N)n1", "三氨基三嗪高氮固化剂，亦可 N-缩水甘油化衍生", True),
    # 潜伏型与催化型固化剂
    PrecursorCore("dicy", "双氰胺 (DICY)", "潜伏型固化剂", "hardener", "N#CN=C(N)N", "CFRP 预浸料标杆潜伏型固化剂，长储存期高温固化", True),
    PrecursorCore("mi_2", "2-甲基咪唑 (2-MI)", "咪唑固化剂", "hardener", "Cc1ncc[nH]1", "经典咪唑催化型固化剂/促进剂，中温快速凝胶", True),
    PrecursorCore("emi_24", "2-乙基-4-甲基咪唑 (2E4MZ)", "咪唑固化剂", "hardener", "CCc1nc(C)c[nH]1", "液态咪唑固化剂，长适用期酸酐/双氰胺促进剂", True),
    PrecursorCore("pz_2", "2-苯基咪唑 (2PZ)", "咪唑固化剂", "hardener", "c1ccc(-c2ncc[nH]2)cc1", "高耐温咪唑固化剂，电工浇注与粉末涂料用", True),
    # 硫醇快固型固化剂
    PrecursorCore("petmp", "季戊四醇四硫醇 (PETMP)", "硫醇固化剂", "hardener", "SCC(CS)(CS)CS", "四官能聚硫醇，低温快速固化与光固化体系", True),
]


# ==============================================================================
# 4. 阶段 1：骨架裂变生成器 (Scaffold Fission & Assembly)
# ==============================================================================

# 星型/四官能组装用"环臂"模板：(酚臂, 胺臂)。片段第一个原子为与中心核的连接点。
# 连接点碳必须在片段首位、羟基/氨基在环上其他位置；若写成 Oc1ccc(...)cc1（氧在前），
# 中心碳会连到氧上形成醚而非 C-C 键（实测 functionality=0）。
RING_ARM_TEMPLATES: dict[str, tuple[str, str]] = {
    "benzene":        ("c1ccc(O)cc1",             "c1ccc(N)cc1"),
    "naphthalene_14": ("c1ccc(O)c2ccccc12",       "c1ccc(N)c2ccccc12"),
    "naphthalene_26": ("c1ccc2cc(O)ccc2c1",       "c1ccc2cc(N)ccc2c1"),
    "anthracene":     ("c1ccc2cc3ccccc3cc2c1O",   "c1ccc2cc3ccccc3cc2c1N"),
    "pyrene":         ("c1cc2ccc3cccc4ccc(O)c1c2c34", "c1cc2ccc3cccc4ccc(N)c1c2c34"),
    "biphenyl_44":    ("c1ccc(-c2ccc(O)cc2)cc1",  "c1ccc(-c2ccc(N)cc2)cc1"),
    "terphenyl":      ("c1ccc(-c2ccc(-c3ccc(O)cc3)cc2)cc1", "c1ccc(-c2ccc(-c3ccc(N)cc3)cc2)cc1"),
    "fluorene":       ("c1ccc2c(c1)Cc1ccc(O)cc12", "c1ccc2c(c1)Cc1ccc(N)cc12"),
    "xanthene":       ("c1ccc2c(c1)Oc1ccc(O)cc1C2", "c1ccc2c(c1)Oc1ccc(N)cc1C2"),
    "carbazole":      ("c1ccc2[nH]c3ccc(O)cc3c2c1", "c1ccc2[nH]c3ccc(N)cc3c2c1"),
    "quinoxaline":    ("c1ccc2nccnc2c1O",          "c1ccc2nccnc2c1N"),
}

# A2 类别配额表：各生成类别的保底占比（替代先到先得截断，经典母核不占配额）
_CATEGORY_QUOTA = {
    "拓扑衍生双酚": 0.18, "拓扑衍生二胺": 0.18,
    "稠环衍生多酚": 0.12, "稠环衍生多胺": 0.12,
    "特种空间刚性母核": 0.12,
    "不对称杂化母核": 0.09,
    "拓扑星型多酚": 0.04, "拓扑星型多胺": 0.04,
    "多元羧酸衍生母核": 0.06,
}


def _apply_category_quota(
    intermediates: list[PrecursorCore], max_total: int
) -> list[PrecursorCore]:
    """按类别配额截断：经典母核恒保留；配额内每类先取；剩余名额按存量回填。

    注意：PrecursorCore 是 @dataclass（eq=True 时不可哈希），判重一律用 core_id 集合，
    严禁 `set(core_mols)`（会 TypeError: unhashable）。
    """
    quota_budget = max(0, max_total - len(PRECURSOR_CATALOG))
    core_ids = {c.core_id for c in PRECURSOR_CATALOG}
    core_mols = [p for p in intermediates if p.core_id in core_ids]
    generated = [p for p in intermediates if p.core_id not in core_ids]
    by_cat: dict[str, list[PrecursorCore]] = {}
    for p in generated:
        by_cat.setdefault(p.category, []).append(p)
    selected: list[PrecursorCore] = []
    for cat, frac in _CATEGORY_QUOTA.items():
        pool = by_cat.get(cat, [])
        take = min(len(pool), int(quota_budget * frac))
        selected.extend(pool[:take])
    # 剩余名额按存量回填（类别顺序保持注册表顺序，保证确定性）
    used = len(selected)
    remaining = quota_budget - used
    for cat in _CATEGORY_QUOTA:
        pool = by_cat.get(cat, [])
        taken = min(len(pool), int(quota_budget * _CATEGORY_QUOTA[cat]))
        extra = min(len(pool) - taken, remaining)
        selected.extend(pool[taken:taken + extra])
        remaining -= extra
        if remaining <= 0:
            break
    # 未列入配额表的类别（如测试类别）全额保留
    for cat, pool in by_cat.items():
        if cat not in _CATEGORY_QUOTA:
            selected.extend(pool)
    return core_mols + selected


def generate_scaffold_intermediates(
    selected_rings: list[str] | None = None,
    selected_linkers: list[str] | None = None,
    selected_r_groups: list[str] | None = None,
    include_phenols: bool = True,
    include_amines: bool = True,
    include_acids: bool = True,
    max_intermediates: int = 50000,
) -> list[PrecursorCore]:
    """Combinatorially assemble ring + linker + R-group scaffolds into multi-functional intermediates."""
    if not RDKIT_AVAILABLE:
        return list(PRECURSOR_CATALOG)

    rings = [r for r in RING_SCAFFOLDS if (not selected_rings or r.ring_id in selected_rings)]
    linkers = [l for l in LINKER_BRIDGES if (not selected_linkers or l.linker_id in selected_linkers)]
    r_groups = [g for g in R_GROUPS if (not selected_r_groups or g.group_id in selected_r_groups)]

    intermediates: list[PrecursorCore] = list(PRECURSOR_CATALOG)
    seen_smiles = {p.smiles for p in intermediates}

    # 环系/桥联选择生效门控：用户取消勾选的环与桥不再参与组装
    selected_ring_set = {r.ring_id for r in rings} if selected_rings else None
    selected_linker_ids = {l.linker_id for l in linkers} if selected_linkers else None

    def _ring_enabled(*keys: str) -> bool:
        if not selected_ring_set:
            return True
        return any(k in selected_ring_set for k in keys)

    def _linker_enabled(key: str) -> bool:
        if not selected_linker_ids:
            return True
        return key in selected_linker_ids

    # 1. 骨架组装模式
    # 1.1 双核组装模型 (Ring1 - Linker - Ring2)，支持不同环的交叉连接与单/稠环多位点
    # 元组末位为对应的 RING_SCAFFOLDS ring_id，用于按用户选择过滤
    ring_templates = [
        ("benzene_14", "苯环-1,4位", "benzene", "c1ccc({link})cc1", "Oc1cc{r}c({lk}c2ccc(O)cc2)cc1", "Nc1cc{r}c({lk}c2ccc(N)cc2)cc1"),
        ("benzene_13", "苯环-1,3位", "benzene", "c1cccc({link})c1", "Oc1cccc{r}c({lk}c2cccc(O)c2)c1", "Nc1cccc{r}c({lk}c2cccc(N)c2)c1"),
        ("toluene", "甲苯基", "toluene", "c1cc(C)cc({link})c1", "Cc1cc(O)ccc1{lk}c1ccc(O)c(C)c1", "Cc1cc(N)ccc1{lk}c1ccc(N)c(C)c1"),
        ("xylene", "二甲苯基", "xylene", "c1c(C)cc(C)c({link})c1", "Cc1cc(C)c(O)cc1{lk}c1cc(O)c(C)cc1C", "Cc1cc(C)c(N)cc1{lk}c1cc(N)c(C)cc1C"),
        ("naphthalene_14", "1,4-萘环", "naphthalene_14", "c1ccc2c({link})cccc2c1", "Oc1ccc2ccccc2c1{lk}c1c(O)ccc2ccccc12", "Nc1ccc2ccccc2c1{lk}c1c(N)ccc2ccccc12"),
        ("naphthalene_15", "1,5-萘环", "naphthalene_15", "c1ccc2c({link})cccc2c1", "Oc1cccc2c(cccc12){lk}c1cccc2c(O)cccc12", "Nc1cccc2c(cccc12){lk}c1cccc2c(N)cccc12"),
        ("naphthalene_26", "2,6-萘环", "naphthalene_26", "c1cc2cc({link})ccc2cc1", "Oc1ccc2cc({lk}c3ccc4cc(O)ccc4c3)ccc2c1", "Nc1ccc2cc({lk}c3ccc4cc(N)ccc4c3)ccc2c1"),
        ("naphthalene_27", "2,7-萘环", "naphthalene_27", "c1cc2ccc({link})cc2cc1", "Oc1ccc2ccc({lk}c3ccc4ccc(O)cc4c3)cc2c1", "Nc1ccc2ccc({lk}c3ccc4ccc(N)cc4c3)cc2c1"),
        ("biphenyl", "4,4'-联苯", "biphenyl_44", "c1ccc(-c2ccc({link})cc2)cc1", "Oc1ccc(-c2ccc({lk}c3ccc(-c4ccc(O)cc4)cc3)cc2)cc1", "Nc1ccc(-c2ccc({lk}c3ccc(-c4ccc(N)cc4)cc3)cc2)cc1"),
        ("cyclohexyl", "环己基", "cyclohexane", "C1CCC({link})CC1", "OC1CCC({lk}C2CCC(O)CC2)CC1", "NC1CCC({lk}C2CCC(N)CC2)CC1"),
        ("adamantane", "金刚烷基", "adamantane", "C1C2CC3CC1CC({link})(C2)C3", "OC1C2CC3CC1CC({lk}C1C4CC5CC1CC(O)(C4)C5)(C2)C3", "NC1C2CC3CC1CC({lk}C1C4CC5CC1CC(N)(C4)C5)(C2)C3"),
        ("mesitylene", "均三甲苯基", "mesitylene", "Cc1cc(C)cc({link})c1",
         "Oc1cc{r}cc(C)c1{lk}c1cc(O)cc(C)c1C", "Nc1cc{r}cc(C)c1{lk}c1cc(N)cc(C)c1C"),
        ("biphenyl_33", "3,3'-联苯", "biphenyl_33", "c1cccc(-c2cccc({link})c2)c1",
         "Oc1cccc(-c2cccc({lk}c3cccc(O)c3)c2)c1", "Nc1cccc(-c2cccc({lk}c3cccc(N)c3)c2)c1"),
    ]

    for r_name, r_desc, ring_key, ring_base, p_tmpl, a_tmpl in ring_templates:
        if not _ring_enabled(ring_key):
            continue
        for linker in linkers:
            if linker.valency != 2:
                continue
            for r_grp in r_groups:
                r_str = f"({r_grp.smiles})" if r_grp.smiles else ""
                
                # 确定连接键 SMILES
                if linker.linker_id == "direct_bond":
                    lk_str = "-"
                elif linker.linker_id == "methylene":
                    lk_str = "C"
                elif linker.linker_id == "isopropylidene":
                    lk_str = "C(C)(C)"
                elif linker.linker_id == "diethyl_methylene":
                    lk_str = "C(CC)(CC)"
                elif linker.linker_id == "hexafluoroisopropylidene":
                    lk_str = "C(C(F)(F)F)(C(F)(F)F)"
                elif linker.linker_id == "sulfone":
                    lk_str = "S(=O)(=O)"
                elif linker.linker_id == "sulfoxide":
                    lk_str = "S(=O)"
                elif linker.linker_id == "ether":
                    lk_str = "O"
                elif linker.linker_id == "thioether":
                    lk_str = "S"
                elif linker.linker_id == "carbonyl":
                    lk_str = "C(=O)"
                elif linker.linker_id == "cyclohexylidene":
                    lk_str = "C1(CCCCC1)"
                elif linker.linker_id == "ester":
                    lk_str = "C(=O)O"
                elif linker.linker_id == "amide":
                    lk_str = "C(=O)N"
                elif linker.linker_id == "phenyl_phosphonate":
                    lk_str = "P(=O)(c1ccccc1)"
                elif linker.linker_id == "siloxane":
                    lk_str = "[Si](C)(C)O[Si](C)(C)"
                else:
                    lk_str = linker.smiles_pattern if linker.smiles_pattern else "C"

                # 连接桥 SMILES 自带的环闭合编号（如环己基叉 C1(CCCCC1)、二环戊二烯桥
                # C1CC2CC1C1CCC21）会与模板在 {lk} 拼接点尚未闭合的环编号冲突，
                # RDKit 会把桥错误稠合到酚环上生成“单酚”假产物（酚羟基数=1，
                # 环氧化后官能度 1 无法交联）；统一重映射到高位编号——模板拼接点
                # 最多开放 1-3 号环，6-9 号保证无碰撞，桥拓扑结构不变。
                lk_str = lk_str.translate(str.maketrans("1234", "6789"))

                # 构建多元酚中间体
                if include_phenols:
                    smi = p_tmpl.replace("{lk}", lk_str).replace("{r}", r_str)
                    try:
                        mol = Chem.MolFromSmiles(smi)
                        if mol is not None:
                            Chem.SanitizeMol(mol)
                            can_smi = Chem.MolToSmiles(mol, canonical=True)
                            if can_smi not in seen_smiles:
                                seen_smiles.add(can_smi)
                                intermediates.append(
                                    PrecursorCore(
                                        core_id=f"gen_diphenol_{hashlib.md5(can_smi.encode()).hexdigest()[:8]}",
                                        name=f"衍生双酚 ({r_desc}-{linker.name}-{r_grp.name})",
                                        category="拓扑衍生双酚",
                                        role="resin",
                                        smiles=can_smi,
                                        description=f"基于 {r_desc} 骨架与 {linker.name} 桥联的组合衍生前驱体",
                                        default_selected=True,
                                    )
                                )
                    except Exception:
                        pass

                # 构建多元芳香胺中间体
                if include_amines:
                    smi_a = a_tmpl.replace("{lk}", lk_str).replace("{r}", r_str)
                    try:
                        mol_a = Chem.MolFromSmiles(smi_a)
                        if mol_a is not None:
                            Chem.SanitizeMol(mol_a)
                            can_smi_a = Chem.MolToSmiles(mol_a, canonical=True)
                            if can_smi_a not in seen_smiles:
                                seen_smiles.add(can_smi_a)
                                intermediates.append(
                                    PrecursorCore(
                                        core_id=f"gen_diamine_{hashlib.md5(can_smi_a.encode()).hexdigest()[:8]}",
                                        name=f"衍生二胺 ({r_desc}-{linker.name}-{r_grp.name})",
                                        category="拓扑衍生二胺",
                                        role="hardener",
                                        smiles=can_smi_a,
                                        description=f"基于 {r_desc} 骨架与 {linker.name} 桥联的衍生芳香二胺",
                                        default_selected=True,
                                    )
                                )
                    except Exception:
                        pass

    # 1.2 稠环与杂环多羟基/多氨基衍生母核 (Naphthalene, Anthracene, Triazine, Pyridine, etc.)
    poly_cores = [
        ("14_dhn", "1,4-萘环", "naphthalene_14", "Oc1ccc{r}(O)c2ccccc12", "Nc1ccc{r}(N)c2ccccc12"),
        ("15_dhn", "1,5-萘环", "naphthalene_15", "Oc1cccc2c(O)cc{r}c12", "Nc1cccc2c(N)cc{r}c12"),
        ("26_dhn", "2,6-萘环", "naphthalene_26", "Oc1cc2cc(O)c{r}cc2cc1", "Nc1cc2cc(N)c{r}cc2cc1"),
        ("27_dhn", "2,7-萘环", "naphthalene_27", "Oc1ccc2cc(O)c{r}cc2c1", "Nc1ccc2cc(N)c{r}cc2c1"),
        ("anthracene", "9,10-蒽环", "anthracene", "Oc1ccc2cc3c(O)ccc{r}c3cc2c1", "Nc1ccc2cc3c(N)ccc{r}c3cc2c1"),
        ("phenanthrene", "菲环", "phenanthrene", "Oc1ccc2c3ccc(O)c{r}c3ccc2c1", "Nc1ccc2c3ccc(N)c{r}c3ccc2c1"),
        ("fluorene", "芴环", "fluorene", "Oc1cc{r}c2c(c1)Cc1ccc(O)cc12", "Nc1cc{r}c2c(c1)Cc1ccc(N)cc12"),
        ("xanthene", "呫吨/氧杂蒽环", "xanthene", "Oc1cc{r}c2c(c1)Oc1cc(O)ccc1C2", "Nc1cc{r}c2c(c1)Oc1cc(N)ccc1C2"),
        ("pyrene", "芘环", "pyrene", "Oc1cc2ccc3cc(O)c4c{r}ccc(c1)c4c3-2", "Nc1cc2ccc3cc(N)c4c{r}ccc(c1)c4c3-2"),
        ("quinoxaline", "喹喔啉环", "quinoxaline", "Oc1nc(O)c2cc{r}ccc2n1", "Nc1c{r}c2nccnc2cc1N"),
        ("terphenyl", "对三联苯", "terphenyl", "Oc1ccc(-c2cc{r}c(-c3ccc(O)cc3)cc2)cc1", "Nc1ccc(-c2cc{r}c(-c3ccc(N)cc3)cc2)cc1"),
        ("pyridine", "吡啶环", "pyridine", "Oc1cc(O)c{r}cn1", "Nc1cc(N)c{r}cn1"),
        ("carbazole", "咔唑环", "carbazole", "Oc1cc{r}c2[nH]c3ccc(O)cc3c2c1", "Nc1cc{r}c2[nH]c3ccc(N)cc3c2c1"),
        ("adamantane", "金刚烷", "adamantane", "OC1C2CC3CC1CC(O)(C2)C3", "NC1C2CC3CC1CC(N)(C2)C3"),
        ("cardo_bpf", "卡基芴", "cardo_bpf",
         "Oc1ccc2c(c1)C(c1ccccc1)(c1ccccc1)c1cc(O)ccc1-2", "Nc1ccc(C2(c3ccccc3)c3ccccc3-c3ccccc32)cc1"),
        ("s_triazine", "对称三嗪", "s_triazine",
         "Oc1nc(O)nc{r}n1", "Nc1nc(N)nc{r}n1"),
    ]
    for c_id, c_name, ring_key, p_tmpl, a_tmpl in poly_cores:
        if not _ring_enabled(ring_key):
            continue
        for r_grp in r_groups:
            r_str = f"({r_grp.smiles})" if r_grp.smiles else ""
            # 固定模板（无 {r} 槽位）只为首个（无取代）R基生成一次，避免重复去重开销
            if "{r}" not in p_tmpl and r_str:
                continue
            p_smi = p_tmpl.replace("{r}", r_str)
            a_smi = a_tmpl.replace("{r}", r_str)
            for s, cat, role, prefix in [(p_smi, "稠环衍生多酚", "resin", "多酚"), (a_smi, "稠环衍生多胺", "hardener", "多胺")]:
                try:
                    m = Chem.MolFromSmiles(s)
                    if m:
                        Chem.SanitizeMol(m)
                        cs = Chem.MolToSmiles(m, canonical=True)
                        if cs not in seen_smiles:
                            seen_smiles.add(cs)
                            intermediates.append(
                                PrecursorCore(
                                    core_id=f"gen_poly_{hashlib.md5(cs.encode()).hexdigest()[:8]}",
                                    name=f"{c_name}{prefix} ({r_grp.name})",
                                    category=cat,
                                    role=role,
                                    smiles=cs,
                                    description=f"基于 {c_name} 的稠环/杂环衍生前驱体",
                                    default_selected=True,
                                )
                            )
                except Exception:
                    pass

    # 2. 三核星型组装模型 (>CH- 次甲基核)：遍历所有已选环骨架的酚/胺臂
    if _linker_enabled("methine_star"):
        for ring_key, (arm_p, arm_a) in RING_ARM_TEMPLATES.items():
            if not _ring_enabled(ring_key):
                continue
            for r_grp in r_groups:
                # R 基替换：苯臂（benzene）模板含 {r} 槽，其他环臂忽略 R 基
                if ring_key == "benzene":
                    r_str = f"({r_grp.smiles})" if r_grp.smiles else ""
                    arm_p_r = f"c1cc{r_str}c(O)cc1"
                    arm_a_r = f"c1cc{r_str}c(N)cc1"
                else:
                    arm_p_r, arm_a_r = arm_p, arm_a
                for s, cat, role, prefix in [
                    (f"C({arm_p_r})({arm_p_r}){arm_p_r}", "拓扑星型多酚", "resin", "三酚"),
                    (f"C({arm_a_r})({arm_a_r}){arm_a_r}", "拓扑星型多胺", "hardener", "三胺"),
                ]:
                    try:
                        m = Chem.MolFromSmiles(s)
                        if m:
                            Chem.SanitizeMol(m)
                            cs = Chem.MolToSmiles(m, canonical=True)
                            if cs not in seen_smiles:
                                seen_smiles.add(cs)
                                intermediates.append(PrecursorCore(
                                    core_id=f"gen_star_{hashlib.md5(cs.encode()).hexdigest()[:8]}",
                                    name=f"星型{prefix} ({ring_key})",
                                    category=cat, role=role, smiles=cs,
                                    description=f"三官能度星型拓扑核心前驱体（{ring_key} 臂）",
                                    default_selected=True))
                    except Exception:
                        pass

    # 3. 空间四官能组装模型 (四苯乙烷/四苯甲烷/卡基芴/螺双二氢茚，由对应环系与星型桥联控制)
    for r_grp in r_groups:
        r_str = f"({r_grp.smiles})" if r_grp.smiles else ""
        spatial_candidates = []
        if _linker_enabled("ethane_tetra"):
            for ring_key, (arm_p, arm_a) in RING_ARM_TEMPLATES.items():
                if not _ring_enabled(ring_key):
                    continue
                spatial_candidates.append(
                    (f"C({arm_p})({arm_p})C({arm_p})({arm_p})", f"四苯乙烷型四酚 ({ring_key})", "resin"))
                spatial_candidates.append(
                    (f"C({arm_a})({arm_a})C({arm_a})({arm_a})", f"四苯乙烷型四胺 ({ring_key})", "hardener"))
        if _linker_enabled("quaternary_carbon"):
            for ring_key, (arm_p, arm_a) in RING_ARM_TEMPLATES.items():
                if not _ring_enabled(ring_key):
                    continue
                spatial_candidates.append(
                    (f"C({arm_p})({arm_p})({arm_p}){arm_p}", f"四苯甲烷型四酚 ({ring_key})", "resin"))
                spatial_candidates.append(
                    (f"C({arm_a})({arm_a})({arm_a}){arm_a}", f"四苯甲烷型四胺 ({ring_key})", "hardener"))
        if _ring_enabled("fluorene"):
            spatial_candidates.extend([
                (f"Oc1cc{r_str}c(C2(c3ccccc3-c3ccccc32)c2ccc(O)cc2)cc1", "卡基双酚芴", "resin"),
                (f"Nc1cc{r_str}c(C2(c3ccccc3-c3ccccc32)c2ccc(N)cc2)cc1", "卡基双胺芴", "hardener"),
            ])
        if _ring_enabled("spirobiindane"):
            spatial_candidates.extend([
                (f"CC1(C)Cc2ccc(O)c{r_str}c2C12CCCc1cc(O)ccc12", "螺双二氢茚双酚", "resin"),
                (f"CC1(C)Cc2ccc(N)c{r_str}c2C12CCCc1cc(N)ccc12", "螺双二氢茚双胺", "hardener"),
            ])
        for s, name_desc, role in spatial_candidates:
            try:
                m = Chem.MolFromSmiles(s)
                if m:
                    Chem.SanitizeMol(m)
                    cs = Chem.MolToSmiles(m, canonical=True)
                    if cs not in seen_smiles:
                        seen_smiles.add(cs)
                        intermediates.append(
                            PrecursorCore(
                                core_id=f"gen_spatial_{hashlib.md5(cs.encode()).hexdigest()[:8]}",
                                name=f"{name_desc} ({r_grp.name})",
                                category="特种空间刚性母核",
                                role=role,
                                smiles=cs,
                                description=f"高耐热/高官能度特种衍生母核",
                                default_selected=True,
                            )
                        )
            except Exception:
                pass

    # 4. 杂化交联不对称多酚/多胺衍生库 (Cross-linked hybrid diphenols/diamines)
    hybrid_linker_map = [
        ("methylene", "亚甲基", "C"),
        ("isopropylidene", "异丙基", "C(C)(C)"),
        ("hexafluoroisopropylidene", "六氟异丙基", "C(C(F)(F)F)(C(F)(F)F)"),
        ("sulfone", "砜基", "S(=O)(=O)"),
        ("ether", "醚键", "O"),
    ]
    if _ring_enabled("benzene", "naphthalene_14"):
        for lk_id, lk_name, lk_str in hybrid_linker_map:
            if not _linker_enabled(lk_id):
                continue
            for r_grp in r_groups:
                r_str = f"({r_grp.smiles})" if r_grp.smiles else ""
                # 苯酚-桥-萘酚
                smi_hybrid_p = f"Oc1cc{r_str}c({lk_str}c2ccc3ccccc3c2O)cc1"
                smi_hybrid_a = f"Nc1cc{r_str}c({lk_str}c2ccc3ccccc3c2N)cc1"
                for s, name_desc, role in [(smi_hybrid_p, f"苯-萘杂化双酚 ({lk_name})", "resin"), (smi_hybrid_a, f"苯-萘杂化双胺 ({lk_name})", "hardener")]:
                    try:
                        m = Chem.MolFromSmiles(s)
                        if m:
                            Chem.SanitizeMol(m)
                            cs = Chem.MolToSmiles(m, canonical=True)
                            if cs not in seen_smiles:
                                seen_smiles.add(cs)
                                intermediates.append(
                                    PrecursorCore(
                                        core_id=f"gen_hyb_{hashlib.md5(cs.encode()).hexdigest()[:8]}",
                                        name=f"{name_desc} ({r_grp.name})",
                                        category="不对称杂化母核",
                                        role=role,
                                        smiles=cs,
                                        description=f"不对称多环芳香衍生母核",
                                        default_selected=True,
                                    )
                                )
                    except Exception:
                        pass

    # 5. 多元羧酸衍生母核 (include_acids)：缩水甘油酯树脂 + 环状酸酐固化剂双路线
    # 邻位/1,2位二酸可经 R07a/R07b 分子内脱水成环酐，间/对位与全取代型走缩水甘油酯路线
    # 门控语义：酸母核的模板骨架为苯环系，仅当用户勾选了苯环/萘等芳环时才生成
    if include_acids and (_ring_enabled("benzene") or _ring_enabled("naphthalene_14") or _ring_enabled("toluene")):
        acid_cores = [
            ("phthalic_gen", "邻苯二甲酸型", "O=C(O)c1cc{r}ccc1C(=O)O"),
            ("isophthalic_gen", "间苯二甲酸型", "O=C(O)c1cc{r}cc(C(=O)O)c1"),
            ("terephthalic_gen", "对苯二甲酸型", "O=C(O)c1ccc(C(=O)O)c{r}c1"),
            ("trimellitic_gen", "偏苯三甲酸型", "O=C(O)c1ccc(C(=O)O)c(C(=O)O){r}c1"),
            ("trimesic_gen", "均苯三甲酸型", "O=C(O)c1cc(C(=O)O){r}cc(C(=O)O)c1"),
            ("pyromellitic_gen", "均苯四甲酸型", "O=C(O)c1cc(C(=O)O){r}c(C(=O)O)cc1C(=O)O"),
            ("naphthalene_diacid", "萘二甲酸型", "O=C(O)c1cc{r}c2cc(C(=O)O)ccc2c1"),
            ("btda_gen", "二苯酮四甲酸型", "O=C(O)c1ccc(C(=O)c2ccc(C(=O)O)c(C(=O)O)c2)cc1C(=O)O"),
            ("thpa_gen", "四氢邻苯二甲酸型", "O=C(O)C1CC=CCC1C(=O)O"),
            ("hpa_gen", "六氢邻苯二甲酸型", "O=C(O)C1CCCCC1C(=O)O"),
        ]
        for c_id, c_name, tmpl in acid_cores:
            for r_grp in r_groups:
                r_str = f"({r_grp.smiles})" if r_grp.smiles else ""
                if "{r}" not in tmpl and r_str:
                    continue
                smi = tmpl.replace("{r}", r_str)
                try:
                    m = Chem.MolFromSmiles(smi)
                    if m:
                        Chem.SanitizeMol(m)
                        cs = Chem.MolToSmiles(m, canonical=True)
                        if cs not in seen_smiles:
                            seen_smiles.add(cs)
                            intermediates.append(
                                PrecursorCore(
                                    core_id=f"gen_acid_{hashlib.md5(cs.encode()).hexdigest()[:8]}",
                                    name=f"{c_name}母核 ({r_grp.name})",
                                    category="多元羧酸衍生母核",
                                    role="resin",
                                    smiles=cs,
                                    description=f"基于 {c_name} 的多元羧酸衍生前驱体（缩水甘油酯/环酐双路线）",
                                    default_selected=True,
                                )
                            )
                except Exception:
                    pass

    # 类别配额截断：替代"先到先得"，保证柱后（稠环/星型/酸）类别不被饿死
    intermediates = _apply_category_quota(intermediates, max_intermediates)

    return intermediates


# ==============================================================================
# 5. 逆合成拆解器与自建库/BigSMILES 前驱体母核提取 (Retrosynthetic Deconstructor)
# ==============================================================================

# 逆合成反应模板：将单体弹头还原/水解为前驱体多元酚、多元胺、多元酸
RETRO_DECONSTRUCTION_TEMPLATES = [
    # 1. 酚缩水甘油醚水解 -> 多元酚: [c]OCC1CO1 -> [c]OH
    ("retro_glycidyl_ether", "[c:1]OCC1CO1>>[c:1][OH]", "酚缩水甘油醚 -> 多元酚"),
    # 2. 缩水甘油胺水解 -> 芳香胺: [c]N(CC1CO1)CC2CO2 -> [c]NH2
    ("retro_glycidyl_amine", "[c:1]N(CC2CO2)CC3CO3>>[c:1][NH2]", "缩水甘油胺 -> 芳香伯胺"),
    # 3. 缩水甘油单胺水解 -> [c]NH2
    ("retro_glycidyl_monoamine", "[c:1]NCC2CO2>>[c:1][NH2]", "缩水甘油单胺 -> 芳香伯胺"),
    # 4. 羧酸缩水甘油酯水解 -> 多元酸: [C](=O)OCC1CO1 -> [C](=O)OH
    ("retro_glycidyl_ester", "[C:1](=O)OCC2CO2>>[C:1](=O)[OH]", "缩水甘油酯 -> 多元羧酸"),
    # 5. 氰酸酯水解 -> 多元酚: [c]OC#N -> [c]OH
    ("retro_cyanate_ester", "[c:1]OC#N>>[c:1][OH]", "氰酸酯 -> 多元酚"),
    # 6. 双马来酰亚胺水解 -> 芳香胺: [c]N1C(=O)C=CC1=O -> [c]NH2
    ("retro_bismaleimide", "[c:1]N1C(=O)C=CC1=O>>[c:1][NH2]", "双马来酰亚胺 -> 芳香伯胺"),
    # 7. 炔丙基醚水解 -> 多元酚: [c]OCC#C -> [c]OH
    ("retro_propargyl", "[c:1]OCC#C>>[c:1][OH]", "炔丙基醚 -> 多元酚"),
    # 8. 苯并噁嗪水解 -> 多元酚: [c]1OCN(c2ccccc2)Cc1 -> [c]1(OH)c1
    ("retro_benzoxazine", "[c:1]1OCN(c2ccccc2)Cc1>>[c:1][OH]", "苯并噁嗪 -> 多元酚"),
]


_COMPILED_RETRO_TEMPLATES: list[tuple[str, Any, str]] | None = None


def _get_compiled_retro_templates() -> list[tuple[str, Any, str]]:
    global _COMPILED_RETRO_TEMPLATES
    if _COMPILED_RETRO_TEMPLATES is None:
        compiled = []
        if RDKIT_AVAILABLE:
            for retro_id, smarts, desc in RETRO_DECONSTRUCTION_TEMPLATES:
                try:
                    rxn = rdChemReactions.ReactionFromSmarts(smarts)
                    if rxn is not None:
                        compiled.append((retro_id, rxn, desc))
                except Exception:
                    pass
        _COMPILED_RETRO_TEMPLATES = compiled
    return _COMPILED_RETRO_TEMPLATES


def deconstruct_monomer_to_precursor_core(smiles_or_mol: str | Any) -> list[str]:
    """Retrosynthetically deconstruct a monomer (epoxy, BMI, CE, etc.) into its precursor core (phenol, amine, acid)."""
    if not RDKIT_AVAILABLE:
        return []

    if isinstance(smiles_or_mol, str):
        try:
            mol = Chem.MolFromSmiles(smiles_or_mol)
        except Exception:
            return []
    else:
        mol = smiles_or_mol

    if mol is None:
        return []

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return []

    generated_cores = set()
    current_mols = [mol]

    for retro_id, rxn, desc in _get_compiled_retro_templates():
        next_mols = []
        for m in current_mols:
            try:
                products = rxn.RunReactants((m,))
                if products:
                    for prod_tuple in products:
                        prod = prod_tuple[0]
                        try:
                            Chem.SanitizeMol(prod)
                            can_smi = Chem.MolToSmiles(prod, canonical=True)
                            if can_smi and "." not in can_smi:
                                generated_cores.add(can_smi)
                                next_mols.append(prod)
                        except Exception:
                            continue
                else:
                    next_mols.append(m)
            except Exception:
                next_mols.append(m)
        current_mols = next_mols

    # 如果分子本身就是多元酚/多元胺（无需逆合成拆解），直接保留
    raw_can_smi = Chem.MolToSmiles(mol, canonical=True)
    if "O" in raw_can_smi or "N" in raw_can_smi:
        generated_cores.add(raw_can_smi)

    return list(generated_cores)


def parse_and_extract_custom_precursors(
    raw_entries: Sequence[str | dict[str, Any]],
    source_name: str = "自建数据库",
) -> tuple[list[PrecursorCore], list[str]]:
    """Parse custom dataset (SMILES, BigSMILES, PubChem exports), deconstruct into precursor cores, and register."""
    if not RDKIT_AVAILABLE:
        return [], ["RDKit不可用"]

    from .bigsmiles_stochastic_graph import looks_like_bigsmiles, sample_bigsmiles_realizations

    logs = [f"🔍 启动自建/外部前驱体解析器：处理来自【{source_name}】的输入数据..."]
    extracted_precursors: list[PrecursorCore] = []
    seen_smiles = set()

    # 预编译判别官能团 SMARTS（避免在每个分子/片段解析时反复编译）
    patt_nh = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")
    patt_phenol = Chem.MolFromSmarts("[cX3][OX2H]")
    patt_acid = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")

    for idx, entry in enumerate(raw_entries):
        if isinstance(entry, dict):
            raw_str = str(entry.get("smiles") or entry.get("SMILES") or entry.get("bigsmiles") or entry.get("BigSMILES") or "").strip()
            item_name = str(entry.get("name") or entry.get("compound_name") or f"{source_name}_单体_{idx+1}").strip()
        else:
            raw_str = str(entry).strip()
            item_name = f"{source_name}_单体_{idx+1}"

        if not raw_str:
            continue

        candidate_smiles_list = []

        # 1. 检查是否为 BigSMILES 格式聚合物/随机低聚物
        if looks_like_bigsmiles(raw_str):
            try:
                sampled = sample_bigsmiles_realizations(raw_str, n_samples=3, min_repeat_units=1, max_repeat_units=2)
                candidate_smiles_list.extend(sampled)
                logs.append(f"🧬 解析 BigSMILES 语法：{raw_str[:30]}... -> 采样提取出 {len(sampled)} 个代表性低聚片段")
            except Exception as e:
                logs.append(f"⚠️ BigSMILES 解析失败 ({raw_str[:25]}...): {e}")
        else:
            # 普通 SMILES
            candidate_smiles_list.append(raw_str)

        # 2. 对每个提取出的分子执行逆合成母核拆解
        for smi in candidate_smiles_list:
            deconstructed_cores = deconstruct_monomer_to_precursor_core(smi)
            if not deconstructed_cores:
                deconstructed_cores = [smi]

            for core_smi in deconstructed_cores:
                try:
                    mol = Chem.MolFromSmiles(core_smi)
                    if mol is None:
                        continue
                    Chem.SanitizeMol(mol)
                    can_smi = Chem.MolToSmiles(mol, canonical=True)
                except Exception:
                    continue

                if can_smi in seen_smiles:
                    continue

                # 尺寸与复杂度防爆过滤：母核重原子数 > 100 或分子量 > 1500 的超大聚合物/树枝状大分子直接跳过
                if mol.GetNumHeavyAtoms() > 100 or Descriptors.MolWt(mol) > 1500.0:
                    continue

                # 按官能团判别角色（不能用 N/O 元素有无判断：醚键 O 会让 ODA/Jeffamine
                # 等二胺被误判为树脂）
                has_amine = bool(patt_nh and mol.HasSubstructMatch(patt_nh))
                has_phenol = bool(patt_phenol and mol.HasSubstructMatch(patt_phenol))
                has_acid = bool(patt_acid and mol.HasSubstructMatch(patt_acid))
                if has_amine and not (has_phenol or has_acid):
                    role = "hardener"
                    cat = "自建提取多胺"
                elif (has_phenol or has_acid) and not has_amine:
                    role = "resin"
                    cat = "自建提取多酚/多酸"
                else:
                    # 氨基酚/醇胺等双活性母核：树脂与固化剂路线均可参与
                    role = "both"
                    cat = "自建提取双活性母核"

                seen_smiles.add(can_smi)
                core_id = f"custom_{hashlib.md5(can_smi.encode()).hexdigest()[:8]}"
                extracted_precursors.append(
                    PrecursorCore(
                        core_id=core_id,
                        name=f"{item_name}_母核",
                        category=cat,
                        role=role,
                        smiles=can_smi,
                        description=f"从自建库 {source_name} 逆合成拆解出的活性母核骨架",
                        default_selected=True,
                    )
                )

    logs.append(f"✨ 自建库逆合成提取完成：共成功萃取/注册 {len(extracted_precursors)} 个前驱体母核")
    return extracted_precursors, logs

@dataclass
class CombinatorialProduct:
    """A strictly validated combinatorial monomer product with clear chemical provenance."""
    product_smiles: str
    role: str  # "resin" | "hardener"
    resin_type: str  # "epoxy" | "amine" | "anhydride" | "bmi" | "cyanate" | "benzoxazine" | "propargyl"
    precursor_id: str
    precursor_name: str
    precursor_smiles: str  # 溯源前驱体母核结构 SMILES
    reaction_id: str
    reaction_name: str
    functionality: int
    equivalent_weight: float  # EEW for resin (g/eq), AHEW for hardener (g/eq)
    molecular_weight: float
    sa_score: float
    heavy_atoms: int
    rotatable_bonds: int
    aromatic_rings: int
    formula: str
    route_ids: list[str] = field(default_factory=list)   # 全部可行路线 id
    route_names: list[str] = field(default_factory=list)  # 全部可行路线名称
    route_steps_min: int = 1                             # 最短路线步数
    route_redundancy: int = 1                            # 路线冗余度（可行路线数）
    green_score_best: float = 0.0                        # 最优绿色评分


def _calculate_sascore(mol) -> float:
    """Calculate Synthetic Accessibility Score (Fast heuristic fallback)."""
    if mol is None:
        return 10.0
    try:
        n_rings = float(rdMolDescriptors.CalcNumRings(mol))
        n_stereo = float(len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)))
        n_heavy = float(mol.GetNumHeavyAtoms())
        sp3_ratio = float(Descriptors.FractionCSP3(mol))
        score = 1.0 + (n_rings * 0.25) + (n_stereo * 0.4) + (n_heavy * 0.03) + (sp3_ratio * 0.5)
        return float(min(10.0, max(1.0, score)))
    except Exception:
        return 3.0


def _count_reactive_sites(mol, role: str, warhead: str) -> tuple[int, str]:
    """Accurately count crosslinking functionality and identify chemical system based on warhead."""
    if mol is None:
        return 0, "unknown"

    patterns = {
        "epoxide": "[O;r3]1[C;r3][C;r3]1",
        "alkyne": "C#C",
        "alkene": "C=C",
        "cyanate": "OC#N",
        "maleimide": "N1C(=O)C=CC1=O",
        # N-H/N-取代苯并噁嗪通用：O-CH2-N-CH2 与芳环稠合的六元环
        "benzoxazine": "[OX2]1[CX4][NX3][CX4][c]2[c]1cccc2",
        "primary_amine": "[NX3;H2;!$(NC=O)]",
        "secondary_amine": "[NX3;H1;!$(NC=O)]",
        "phenol_oh": "[cX3][OX2H]",
        "thiol": "[SX2H]",
    }

    # 1. 环氧树脂体系
    patt_ep = Chem.MolFromSmarts(patterns["epoxide"])
    if patt_ep and mol.HasSubstructMatch(patt_ep):
        count = len(mol.GetSubstructMatches(patt_ep))
        return count, "epoxy"

    # 2. 双马来酰亚胺 BMI
    patt_bmi = Chem.MolFromSmarts(patterns["maleimide"])
    if patt_bmi and mol.HasSubstructMatch(patt_bmi):
        count = len(mol.GetSubstructMatches(patt_bmi))
        return count, "bmi"

    # 3. 氰酸酯 CE
    patt_ce = Chem.MolFromSmarts(patterns["cyanate"])
    if patt_ce and mol.HasSubstructMatch(patt_ce):
        count = len(mol.GetSubstructMatches(patt_ce))
        return count, "cyanate"

    # 4. 苯并噁嗪 BOZ
    patt_boz = Chem.MolFromSmarts(patterns["benzoxazine"])
    if patt_boz and mol.HasSubstructMatch(patt_boz):
        count = len(mol.GetSubstructMatches(patt_boz))
        return count, "benzoxazine"

    # 5. 炔丙基
    patt_alk = Chem.MolFromSmarts(patterns["alkyne"])
    if patt_alk and mol.HasSubstructMatch(patt_alk):
        count = len(mol.GetSubstructMatches(patt_alk))
        return count, "propargyl"

    # 6. 固化剂体系：酸酐、多胺、多元酚、硫醇、咪唑
    hardener_tokens = ("胺", "酸酐", "酚", "硫醇", "咪唑", "活性氢")
    if role == "hardener" or any(t in warhead for t in hardener_tokens):
        # 酸酐：1 个酸酐基团对应 1 个环氧基（1:1 化学计量），兼容芳构化表示
        n_anh = count_anhydride_groups(mol)
        if n_anh > 0:
            return n_anh, "anhydride"

        total_active_h = 0
        patt_p_nh = Chem.MolFromSmarts(patterns["primary_amine"])
        if patt_p_nh:
            total_active_h += len(mol.GetSubstructMatches(patt_p_nh)) * 2
        patt_s_nh = Chem.MolFromSmarts(patterns["secondary_amine"])
        if patt_s_nh:
            total_active_h += len(mol.GetSubstructMatches(patt_s_nh)) * 1

        if total_active_h > 0:
            return total_active_h, "amine"

        # 酚羟基型固化剂（酚醛/多元酚）：按酚羟基数
        patt_ph = Chem.MolFromSmarts(patterns["phenol_oh"])
        if patt_ph:
            n_ph = len(mol.GetSubstructMatches(patt_ph))
            if n_ph > 0:
                return n_ph, "phenol"

        # 硫醇型固化剂：按巯基数
        patt_sh = Chem.MolFromSmarts(patterns["thiol"])
        if patt_sh:
            n_sh = len(mol.GetSubstructMatches(patt_sh))
            if n_sh > 0:
                return n_sh, "thiol"

        # 咪唑催化型固化剂：按咪唑环数（催化机制，1 个即可启动）
        patt_im1 = Chem.MolFromSmarts("n1cc[nH]c1")
        patt_im2 = Chem.MolFromSmarts("n1cncc1")
        n_im = 0
        if patt_im1:
            n_im += len(mol.GetSubstructMatches(patt_im1))
        if patt_im2:
            n_im += len(mol.GetSubstructMatches(patt_im2))
        if n_im > 0:
            return n_im, "imidazole"

    return 0, "other"


def calculate_stoichiometry(mol, role: str, warhead: str) -> dict[str, Any]:
    """Calculate equivalent weight (EEW/AHEW), functionality, MW and formula."""
    if mol is None:
        return {"functionality": 0, "equivalent_weight": 0.0, "resin_type": "unknown", "mw": 0.0}

    mw = float(Descriptors.MolWt(mol))
    func, sys_type = _count_reactive_sites(mol, role, warhead)

    eq_weight = round(mw / max(1, func), 2) if func > 0 else 0.0
    return {
        "functionality": func,
        "equivalent_weight": eq_weight,
        "resin_type": sys_type,
        "mw": round(mw, 2),
    }


def _apply_reaction_exhaustively(mol, rxn) -> list:
    """Apply a Reaction SMARTS to all available reaction sites efficiently with canonical deduplication.

    语义约定：只返回真正发生反应的产物（完全取代终产物）。
    - 底物若不匹配该反应，返回空列表（绝不允许未反应底物冒充反应产物入库）；
    - 每轮只保留反应过的分子，部分取代的中间体被丢弃，确保官能度统计准确。
    """
    current_mols = [mol]
    seen_in_expansion = set()
    reacted_ever = False

    for _ in range(4):
        next_mols = []
        for m in current_mols:
            try:
                products = rxn.RunReactants((m,))
            except Exception:
                continue
            for prod_tuple in products:
                prod = prod_tuple[0]
                try:
                    Chem.SanitizeMol(prod)
                    can_smi = Chem.MolToSmiles(prod, canonical=True)
                    if can_smi in seen_in_expansion:
                        continue
                    seen_in_expansion.add(can_smi)
                    next_mols.append(prod)
                    reacted_ever = True
                except Exception:
                    continue
        if not next_mols:
            break
        # 分支剪枝：单体衍生只需完全取代产物，每轮保留前 24 个代表性分子，阻断多位点组合爆炸
        current_mols = next_mols[:24]

    return current_mols if reacted_ever else []


_SOUNDNESS_SMARTS_CACHE: dict[str, Any] = {}


def _get_soundness_patt(smarts: str):
    if not RDKIT_AVAILABLE or not smarts:
        return None
    if smarts in _SOUNDNESS_SMARTS_CACHE:
        return _SOUNDNESS_SMARTS_CACHE[smarts]
    p = Chem.MolFromSmarts(smarts)
    _SOUNDNESS_SMARTS_CACHE[smarts] = p
    return p


def validate_monomer_chemical_soundness(mol, role: str = "resin") -> tuple[bool, str]:
    """严格校验单体化学相容性与稳定性（严禁互反应基团共存导致自分子交联聚合）。

    解决用户痛点：「某些分子设计的不合理，比如同一分子里面有环氧和酚羟基」。

    物理化学原则：
    1. 环氧树脂单体 (role=resin 或含环氧基)：
       - 严禁游离酚羟基 [c][OX2H]（发生自加成自固化，属于反应不完全或烯烃直接环氧化导致的自交联缺陷结构）；
       - 严禁游离羧基、强酸基（催化开环失活）；
       - 严禁未保护的伯/仲胺基（室温自分子固化交联成胶）；
       - 严禁巯基、环状酸酐基。
    2. 固化剂单体 (role=hardener)：
       - 严禁带有环氧基（角色混乱且自分子反应）；
       - 酸酐固化剂严禁带游离胺基或酚/醇羟基（分子内自开环失效）；
       - 胺类固化剂严禁带游离羧酸（酸碱盐析中和失效）。
    3. 通用高危结构：
       - 严禁过氧键 (-O-O-)、易水解发烟酰卤 (-COX)。
    """
    if mol is None:
        return False, "分子对象为空"
    if not RDKIT_AVAILABLE:
        return True, "RDKit不可用，跳过相容性校验"

    role_str = str(role or "resin").strip().lower()

    # 通用高危过滤：过氧键与酰卤
    p_perox = _get_soundness_patt("[OX2,OX1-][OX2,OX1-]")
    if p_perox and mol.HasSubstructMatch(p_perox):
        return False, "含有易分解过氧键 (-O-O-)"

    p_acyl = _get_soundness_patt("[CX3](=[OX1])[F,Cl,Br,I]")
    if p_acyl and mol.HasSubstructMatch(p_acyl):
        return False, "含有易水解发烟酰卤基团 (-COX)"

    p_ep = _get_soundness_patt("C1OC1")
    n_epoxy = len(mol.GetSubstructMatches(p_ep)) if p_ep else 0

    p_ph = _get_soundness_patt("[c][OX2H]")
    n_phenol = len(mol.GetSubstructMatches(p_ph)) if p_ph else 0

    p_acid = _get_soundness_patt("[CX3](=O)[OX2H]")
    n_acid = len(mol.GetSubstructMatches(p_acid)) if p_acid else 0

    p_amine = _get_soundness_patt("[NX3;H2,H1;!$(NC=O)]")
    n_amine = len(mol.GetSubstructMatches(p_amine)) if p_amine else 0

    p_thiol = _get_soundness_patt("[SX2H]")
    n_thiol = len(mol.GetSubstructMatches(p_thiol)) if p_thiol else 0

    n_anh = count_anhydride_groups(mol)

    # 1. 环氧树脂体系单体（含环氧基或角色为 resin）
    if n_epoxy > 0 or role_str == "resin":
        if n_epoxy > 0:
            if n_phenol > 0:
                return False, f"分子内同时存在环氧基({n_epoxy})与游离酚羟基({n_phenol})，存在自交联聚合缺陷，无法作为稳定单体存储"
            if n_acid > 0:
                return False, f"分子内同时存在环氧基({n_epoxy})与游离羧基({n_acid})，易催化开环自交联"
            if n_amine > 0:
                return False, f"分子内同时存在环氧基({n_epoxy})与游离胺基({n_amine})，室温即发生自分子交联"
            if n_thiol > 0:
                return False, f"分子内同时存在环氧基({n_epoxy})与巯基({n_thiol})，易发生自加成失效"
            if n_anh > 0:
                return False, f"分子内同时存在环氧基({n_epoxy})与酸酐基({n_anh})，属于互反应自固化混合体"

    # 2. 固化剂体系单体
    if role_str == "hardener":
        if n_epoxy > 0:
            return False, f"固化剂角色单体不得含有环氧基({n_epoxy})"
        if n_anh > 0 and (n_amine > 0 or n_phenol > 0):
            return False, "酸酐固化剂分子内不得含有游离胺基或酚羟基（分子内自开环失效）"
        if n_amine > 0 and n_acid > 0:
            return False, "多元胺固化剂分子内不得含有游离羧基（酸碱盐析中和失效）"

    return True, "化学结构自洽"


def _precursor_matches_reaction(parent_mol, rxn_def: SyntheticReaction | None, rxn, r_templates=None) -> bool:
    """实测底物与反应模板的正交性匹配（避免不合理反应打靶）。"""
    if parent_mol is None:
        return False

    rxn_id = str(getattr(rxn_def, "reaction_id", "") or "")

    # [前置正交性约束]
    # 双键环氧化 (R04) 必须要求底物上没有未反应的游离酚羟基、游离芳胺或羧基。
    # 否则过氧酸直接氧化烯丙基双键会产生「环氧 + 游离酚羟基」的自交联怪胎！
    if rxn_id == "olefin_epoxidation":
        p_ph = _get_soundness_patt("[c][OX2H]")
        if p_ph and parent_mol.HasSubstructMatch(p_ph):
            return False
        p_am = _get_soundness_patt("[c][NX3;H2]")
        if p_am and parent_mol.HasSubstructMatch(p_am):
            return False
        p_acid = _get_soundness_patt("[CX3](=O)[OX2H]")
        if p_acid and parent_mol.HasSubstructMatch(p_acid):
            return False

    templates = r_templates if r_templates is not None else (rxn.GetReactants() if rxn is not None else ())
    try:
        for reactant_template in templates:
            if (
                reactant_template is not None
                and reactant_template.GetNumAtoms() > 0
                and parent_mol.HasSubstructMatch(reactant_template)
            ):
                return True
    except Exception:
        pass
    return False


def _min_functionality_for_type(resin_type: str, min_functionality: int) -> int:
    """按化学体系放宽官能度下限：
    - 单酸酐固化剂（MTHPA/HHPA/NMA 型）是工业标准品，1 个酸酐基对应 1 个环氧基；
    - 咪唑为催化型固化剂，1 个咪唑环即可启动固化；
    其余体系（环氧/胺/酚/硫醇等）维持用户设定下限。
    """
    if resin_type in ("anhydride", "imidazole"):
        return 1
    return min_functionality


def _make_product_record(
    mol,
    can_smi: str,
    role: str,
    resin_type: str,
    precursor: PrecursorCore,
    reaction_id: str,
    reaction_name: str,
    functionality: int,
    equivalent_weight: float,
    mw: float,
    sa: float,
    route: SyntheticRoute | None = None,
) -> CombinatorialProduct:
    return CombinatorialProduct(
        product_smiles=can_smi,
        role=role,
        resin_type=resin_type,
        precursor_id=precursor.core_id,
        precursor_name=precursor.name,
        precursor_smiles=precursor.smiles,
        reaction_id=reaction_id,
        reaction_name=reaction_name,
        functionality=functionality,
        equivalent_weight=equivalent_weight,
        molecular_weight=mw,
        sa_score=round(sa, 2),
        heavy_atoms=int(mol.GetNumHeavyAtoms()),
        rotatable_bonds=int(Descriptors.NumRotatableBonds(mol)),
        aromatic_rings=int(Descriptors.NumAromaticRings(mol)),
        formula=rdMolDescriptors.CalcMolFormula(mol),
        route_ids=[route.route_id] if route else [],
        route_names=[route.name] if route else [],
        route_steps_min=len(route.steps) if route else 1,
        route_redundancy=1,
        green_score_best=route.green_score if route else 0.0,
    )


_ALLOWED_ELEMENTS_ROUTE = frozenset(('C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si'))


def _strip_atom_maps(mol):
    """剥离原子映射编号（仅限内部位点追踪使用，绝不进入产物 SMILES/记录字段）。"""
    stripped = Chem.RWMol(mol)
    for atom in stripped.GetAtoms():
        if atom.GetAtomMapNum():
            atom.SetAtomMapNum(0)
    return stripped.GetMol()


def _label_step_entry_atoms(mol):
    """给进入路线步的分子全部原子打上顺序映射编号，用于跨轮追踪"原始位点"。

    编号仅在本步穷举内部存活：产物返回前由 _strip_atom_maps 全部剥离。
    """
    labeled = Chem.RWMol(mol)
    for i, atom in enumerate(labeled.GetAtoms()):
        atom.SetAtomMapNum(i + 1)
    return labeled.GetMol()


def _consumed_template_positions(rxn, reactant_template) -> tuple[int, ...]:
    """反应物模板中"被消耗原子"的查询位置：其映射编号不再出现在产物侧模板。

    这些原子正是每次反应应用真正从底物移除的原子（如酚羟基氧/氯代氯），
    "原始位点"判定只检查它们；模板不消耗原子时（如烯烃环氧化）返回空元组。
    """
    try:
        product_maps = set()
        for t in rxn.GetProducts():
            for atom in t.GetAtoms():
                m = atom.GetAtomMapNum()
                if m > 0:
                    product_maps.add(m)
        return tuple(
            pos for pos, atom in enumerate(reactant_template.GetAtoms())
            if atom.GetAtomMapNum() > 0 and atom.GetAtomMapNum() not in product_maps
        )
    except Exception:
        return ()


def _count_original_site_matches(mol, reactant_template, consumed_positions) -> int:
    """统计分子上"原始位点"数：位点匹配中被消耗位置的原子仍带本步进入时的映射编号。

    反应新建的原子不带编号（映射号为 0），据此区分"进入本步时已有的位点"
    与"本步反应新建的位点"。
    """
    if mol is None or reactant_template is None:
        return 0
    try:
        matches = mol.GetSubstructMatches(reactant_template)
    except Exception:
        return 0
    if not consumed_positions:
        return len(matches)
    count = 0
    for match in matches:
        try:
            if all(mol.GetAtomWithIdx(match[p]).GetAtomMapNum() > 0 for p in consumed_positions):
                count += 1
        except Exception:
            continue
    return count


def _apply_route_step_exhaustively(mol, rxn) -> list:
    """路线链单步反应的有界穷举执行器（≤4 轮，语义对齐平铺模板路径）。

    与 _apply_reaction_exhaustively 共用同一套穷举思想（反复应用直至无新产物、
    canonical 去重、只返回真正反应过的分子、底物不匹配返回空列表），多位点
    底物（双酚A 的两个酚羟基）由此逐轮到达全取代产物 DGEBA，官能度不再虚低。

    不直接调用 _apply_reaction_exhaustively 的原因：氯代醇等步骤的 SMARTS 产物
    自身仍带可反应羟基，"无新产物即停"的终止条件永不满足——4 轮内会全部长成
    过度烷基化自聚副产物，既丢掉真实中间体（氯代醇），又生成官能度虚高的假
    产物。故本执行器先用 _label_step_entry_atoms 标记进入本步时的既有位点
    （原始位点），每轮只保留"恰好消耗一个原始位点"的产物：原有位点逐轮全
    取代，新建位点上的自聚链被拒绝。产物返回前剥离全部映射编号。
    """
    reactant_template = None
    try:
        for t in rxn.GetReactants():
            if t is not None and t.GetNumAtoms() > 0:
                reactant_template = t
                break
    except Exception:
        return []
    if reactant_template is None:
        return []

    consumed_positions = _consumed_template_positions(rxn, reactant_template)
    current_mols = [_label_step_entry_atoms(mol)]
    seen_in_expansion = set()
    reacted_ever = False

    for _ in range(4):
        next_mols = []
        for m in current_mols:
            try:
                observed = rxn.RunReactants((m,))
            except Exception:
                continue
            n_orig = _count_original_site_matches(m, reactant_template, consumed_positions)
            for prod_tuple in observed:
                prod = prod_tuple[0]
                try:
                    Chem.SanitizeMol(prod)
                except Exception:
                    continue
                # 守卫：只有消耗了一个原始位点的产物才允许进入下一轮/下一步；
                # 消耗新建位点的过度生长链在此被拒绝（n_orig 不减）。
                if _count_original_site_matches(prod, reactant_template, consumed_positions) != n_orig - 1:
                    continue
                try:
                    dedup_key = Chem.MolToSmiles(_strip_atom_maps(prod), canonical=True)
                except Exception:
                    continue
                if dedup_key in seen_in_expansion:
                    continue
                seen_in_expansion.add(dedup_key)
                next_mols.append(prod)
                reacted_ever = True
        if not next_mols:
            break
        # 分支剪枝：每轮保留前 24 个位点反应分子，避免四酚/星型三酚多位点组合膨胀
        current_mols = next_mols[:24]

    if not reacted_ever:
        return []
    return [_strip_atom_maps(m) for m in current_mols]


def _expand_route_chain(
    parent_mol,
    route: SyntheticRoute,
    precursor: PrecursorCore,
    mw_range: tuple[float, float],
    min_functionality: int = 2,
    compiled_steps: list[Any] | None = None,
) -> list[CombinatorialProduct]:
    """沿多步路线顺序执行 Reaction SMARTS。

    - 每一步均为有界穷举执行（见 _apply_route_step_exhaustively）：多位点底物
      （如双酚A 两个酚羟基）在每一步内逐轮全取代，与平铺模板路径的
      _apply_reaction_exhaustively 语义对齐，末产物官能度不再虚低；
    - 中间步骤产物 → role="intermediate" 入库（无官能度/当量需求，不进配方筛选）；
    - 最后一步产物 → 以 route.target_role 入库并计算 EEW/AHEW；
    - SA 分数豁免是刻意设计：完整合成路线本身已证明可合成性；但官能度门禁
      仍然生效——热固网络物理上要求 ≥2 个可反应基团，单官能产物无法交联；
    - 任意一步不匹配 → 不再产出末步产物（已完成步骤的中间体仍会入库）；未反应底物绝不冒充终产物。
    """
    from .task_manager import is_cancelled

    results: list[CombinatorialProduct] = []
    if not route.steps:
        return results

    if compiled_steps is not None and len(compiled_steps) == len(route.steps):
        compiled = compiled_steps
    else:
        compiled = []
        for step in route.steps:
            if not step.reaction_smarts:
                return results
            try:
                rxn = rdChemReactions.ReactionFromSmarts(step.reaction_smarts)
            except Exception:
                return results
            if rxn is None:
                # ReactionFromSmarts 对非法 SMARTS 返回 None 而非抛异常，须在此显式拦截、快速失败
                return results
            compiled.append(rxn)

    current_mols = [parent_mol]
    seen_inter = set()

    for idx, (step, rxn) in enumerate(zip(route.steps, compiled)):
        if is_cancelled():
            return results
        next_mols = []
        seen_step = set()
        for m in current_mols:
            # 单步有界穷举（对齐平铺路径 _apply_reaction_exhaustively 的语义）：
            # 多位点底物（如双酚A 的两个酚羟基）必须逐轮全取代，否则只能得到
            # 官能度 1 的单取代产物、被末步门禁整体拒收，路线列永远为空。
            # 底物不匹配时返回空列表，绝不会让未反应底物冒充产物入库。
            try:
                step_products = _apply_route_step_exhaustively(m, rxn)
            except Exception:
                continue
            for prod in step_products:
                can = Chem.MolToSmiles(prod, canonical=True)
                if "." in can:
                    continue
                if any(atom.GetSymbol() not in _ALLOWED_ELEMENTS_ROUTE for atom in prod.GetAtoms()):
                    continue
                if can in seen_step:
                    continue
                seen_step.add(can)
                next_mols.append(prod)
        if not next_mols:
            return results          # 断链：路线不成立

        is_last = idx == len(route.steps) - 1
        for prod in next_mols:
            can = Chem.MolToSmiles(prod, canonical=True)
            if is_last:
                # 末步门槛：官能度门禁与平铺模板路径一致（_min_functionality_for_type 按
                # resin_type 类型感知放宽）；SA 分数上限豁免保留——完整合成路线本身已
                # 证明可合成性。官能度非启发式，而是热固网络交联的物理前提，故不可豁免。
                stoich = calculate_stoichiometry(prod, route.target_role, "环氧")
                min_req = _min_functionality_for_type(stoich["resin_type"], min_functionality)
                if stoich["functionality"] < min_req:
                    continue
                if not (mw_range[0] <= stoich["mw"] <= mw_range[1]):
                    continue
                # [化学相容性门禁] 路线链末步产物必须是官能团自洽的稳定单体
                is_sound, _ = validate_monomer_chemical_soundness(prod, route.target_role)
                if not is_sound:
                    continue
                results.append(_make_product_record(
                    prod, can, route.target_role, stoich["resin_type"], precursor,
                    route.route_id, route.name, stoich["functionality"],
                    stoich["equivalent_weight"], stoich["mw"],
                    _calculate_sascore(prod), route=route))
            else:
                if can in seen_inter:
                    continue
                seen_inter.add(can)
                results.append(CombinatorialProduct(
                    product_smiles=can, role="intermediate", resin_type="intermediate",
                    precursor_id=precursor.core_id, precursor_name=precursor.name,
                    precursor_smiles=precursor.smiles,
                    reaction_id=step.reaction_id,
                    reaction_name=f"{route.route_id} 中间体",
                    functionality=0, equivalent_weight=0.0,
                    molecular_weight=float(Descriptors.MolWt(prod)),
                    sa_score=round(_calculate_sascore(prod), 2),
                    heavy_atoms=int(prod.GetNumHeavyAtoms()),
                    rotatable_bonds=int(Descriptors.NumRotatableBonds(prod)),
                    aromatic_rings=int(Descriptors.NumAromaticRings(prod)),
                    formula=rdMolDescriptors.CalcMolFormula(prod),
                    route_ids=[route.route_id], route_names=[route.name],
                    route_steps_min=idx + 1, route_redundancy=1,
                    green_score_best=route.green_score,
                ))
        current_mols = next_mols[:24]

    return results


def _process_single_precursor_reactions(
    precursor: PrecursorCore,
    compiled_reactions: list[tuple[SyntheticReaction, Any]],
    reactions: list[SyntheticReaction],
    min_functionality: int,
    max_sa_score: float,
    mw_range: tuple[float, float],
    enable_route_chains: bool = True,
    route_chain_ids: list[str] | None = None,
    compiled_routes: list[tuple[SyntheticRoute, list[Any]]] | None = None,
) -> list[CombinatorialProduct]:
    """Worker function for parallel multi-core reaction expansion with cancellation support."""
    from .task_manager import is_cancelled
    if is_cancelled():
        return []

    local_results: list[CombinatorialProduct] = []
    seen = set()

    try:
        parent_mol = Chem.MolFromSmiles(precursor.smiles)
        if parent_mol is None:
            return []
        Chem.SanitizeMol(parent_mol)
    except Exception:
        return []

    # 前置物理化学可行性过滤：
    # 1. 若母核分子量已超上限 5% (mw_range[1] * 1.05)，接枝衍生只会更重，绝不可能进入合格区间，直接跳过
    parent_mw = Descriptors.MolWt(parent_mol)
    if parent_mw > mw_range[1] * 1.05:
        return []
    # 2. 限制超大重原子骨架（>120 个重原子），避免多反应位点下 2^N 级组合爆炸
    if parent_mol.GetNumHeavyAtoms() > 120:
        return []

    # 1. 天然固化剂母核直接继承 (R12) —— 多胺/酸酐/双氰胺/咪唑/硫醇类母核原样入库
    if precursor.role in ("hardener", "both") and any(r.reaction_id == "native_hardener_retain" for r in reactions):
        stoich = calculate_stoichiometry(parent_mol, "hardener", precursor.category)
        min_req = _min_functionality_for_type(stoich["resin_type"], min_functionality)
        if (
            stoich["functionality"] >= min_req
            and mw_range[0] <= stoich["mw"] <= mw_range[1]
        ):
            can_smi = Chem.MolToSmiles(parent_mol, canonical=True)
            if can_smi not in seen:
                seen.add(can_smi)
                sa = _calculate_sascore(parent_mol)
                if sa <= max_sa_score:
                    local_results.append(
                        _make_product_record(
                            parent_mol, can_smi, "hardener", stoich["resin_type"], precursor,
                            "native_hardener_retain", "天然固化剂母核直接继承",
                            stoich["functionality"], stoich["equivalent_weight"], stoich["mw"], sa,
                        )
                    )

    # 2. 多元酚/酚醛固化剂直接继承 (R14) —— 酚羟基型固化剂路线
    if precursor.role in ("resin", "both") and any(r.reaction_id == "phenolic_hardener_retain" for r in reactions):
        stoich_ph = calculate_stoichiometry(parent_mol, "hardener", "多元酚羟基固化剂")
        if (
            stoich_ph["resin_type"] == "phenol"
            and stoich_ph["functionality"] >= min_functionality
            and mw_range[0] <= stoich_ph["mw"] <= mw_range[1]
        ):
            can_smi = Chem.MolToSmiles(parent_mol, canonical=True)
            if can_smi not in seen:
                seen.add(can_smi)
                sa = _calculate_sascore(parent_mol)
                if sa <= max_sa_score:
                    local_results.append(
                        _make_product_record(
                            parent_mol, can_smi, "hardener", "phenol", precursor,
                            "phenolic_hardener_retain", "多元酚/酚醛固化剂直接继承",
                            stoich_ph["functionality"], stoich_ph["equivalent_weight"], stoich_ph["mw"], sa,
                        )
                    )

    # 3. 遍历多通道合成反应进行衍生
    for item in compiled_reactions:
        if is_cancelled():
            break

        rxn_def = item[0]
        rxn = item[1]
        r_templates = item[2] if len(item) > 2 else None

        # 预过滤：用反应物侧 SMARTS 实测底物能否反应（取代按类别字符串猜测，
        # 避免脂肪胺被芳香模板放行后原样穿透成“假树脂”）
        if not _precursor_matches_reaction(parent_mol, rxn_def, rxn, r_templates):
            continue

        try:
            candidate_mols = _apply_reaction_exhaustively(parent_mol, rxn)
        except Exception:
            continue

        for prod_mol in candidate_mols:
            if is_cancelled():
                break

            try:
                Chem.SanitizeMol(prod_mol)
                canonical_smiles = Chem.MolToSmiles(prod_mol, canonical=True)
            except Exception:
                continue

            if canonical_smiles in seen:
                continue

            # 质量门禁校验 (Quality Gate)
            if any(atom.GetAtomicNum() not in ALLOWED_ELEMENTS for atom in prod_mol.GetAtoms()):
                continue
            if "." in canonical_smiles:
                continue

            target_role = rxn_def.target_role if rxn_def.target_role != "both" else precursor.role
            stoich = calculate_stoichiometry(prod_mol, target_role, rxn_def.expected_warhead)
            min_req = _min_functionality_for_type(stoich["resin_type"], min_functionality)
            if stoich["functionality"] < min_req:
                continue

            if not (mw_range[0] <= stoich["mw"] <= mw_range[1]):
                continue

            # [化学相容性门禁] 严禁互反应基团共存（如环氧-酚羟基、环氧-胺基等自聚合自交联缺陷）
            is_sound, _ = validate_monomer_chemical_soundness(prod_mol, target_role)
            if not is_sound:
                continue

            sa = _calculate_sascore(prod_mol)
            if sa > max_sa_score:
                continue

            seen.add(canonical_smiles)
            local_results.append(
                CombinatorialProduct(
                    product_smiles=canonical_smiles,
                    role=target_role,
                    resin_type=stoich["resin_type"],
                    precursor_id=precursor.core_id,
                    precursor_name=precursor.name,
                    precursor_smiles=precursor.smiles,
                    reaction_id=rxn_def.reaction_id,
                    reaction_name=rxn_def.name,
                    functionality=stoich["functionality"],
                    equivalent_weight=stoich["equivalent_weight"],
                    molecular_weight=stoich["mw"],
                    sa_score=round(sa, 2),
                    heavy_atoms=int(prod_mol.GetNumHeavyAtoms()),
                    rotatable_bonds=int(Descriptors.NumRotatableBonds(prod_mol)),
                    aromatic_rings=int(Descriptors.NumAromaticRings(prod_mol)),
                    formula=rdMolDescriptors.CalcMolFormula(prod_mol),
                )
            )

    # 4. 路线链多步展开 (P1-P4 羟基环氧化等)：中间体 role=intermediate 入库
    if enable_route_chains:
        if compiled_routes is not None:
            for route, c_steps in compiled_routes:
                if route_chain_ids is not None and route.route_id not in route_chain_ids:
                    continue
                chain_prods = _expand_route_chain(
                    parent_mol, route, precursor, mw_range, min_functionality, compiled_steps=c_steps
                )
                for cp in chain_prods:
                    local_results.append(cp)
        else:
            for route in ROUTE_CHAINS:
                if route_chain_ids is not None and route.route_id not in route_chain_ids:
                    continue
                chain_prods = _expand_route_chain(parent_mol, route, precursor, mw_range, min_functionality)
                for cp in chain_prods:
                    # 与单步模板产物撞车也不丢：路线累加由调用方 _merge_product_records 完成
                    local_results.append(cp)

    return local_results


# ==============================================================================
# 6. 全流水线正交展开与配方级高通量衔接引擎 (多核并行加速)
# ==============================================================================

def _merge_product_records(
    results: list[CombinatorialProduct],
    seen_idx: dict[str, int],
    item: CombinatorialProduct,
) -> bool:
    """按 canonical SMILES 合并产物：同产物累加路线，不丢弃先行命中的路线信息。
    返回 bool 表示是否新增了一个非 intermediate 的可用单体（供 O(1) 实时配额统计）。
    """
    key = item.product_smiles
    idx = seen_idx.get(key)
    if idx is None:
        seen_idx[key] = len(results)
        results.append(item)
        return item.role != "intermediate"
    existing = results[idx]
    for rid in item.route_ids:
        if rid not in existing.route_ids:
            existing.route_ids.append(rid)
    for rn in item.route_names:
        if rn not in existing.route_names:
            existing.route_names.append(rn)
    existing.route_redundancy = max(1, len(existing.route_ids))
    existing.route_steps_min = min(existing.route_steps_min, item.route_steps_min)
    existing.green_score_best = max(existing.green_score_best, item.green_score_best)
    return False


def _serialize_route_columns(route_ids, route_names) -> tuple[str, str]:
    """路线 id/名称成对按 route_id 排序后序列化为分号字符串，保证跨运行字节级可复现。

    并发模式下 _merge_product_records 按线程完成顺序累加路线，到达顺序不可控；
    序列化时统一按 route_id 排序即可消除该不确定性。id 与名称始终成对排序，
    绝不破坏 route_ids/route_names 两列的索引对应关系。
    防御性对齐：id 与名称在合并时独立去重，极端情况下长度可能不一致；
    此时缺失名称回退为 route_id 本身，避免出现“有 id 无名称”的空档，
    且对任何输入都不抛异常。
    """
    ids = route_ids if isinstance(route_ids, list) else []
    names = route_names if isinstance(route_names, list) else []
    pairs = []
    for i, rid in enumerate(ids):
        name = names[i] if i < len(names) else None
        pairs.append((str(rid), str(name) if name else str(rid)))
    pairs.sort(key=lambda p: p[0])
    return ";".join(p[0] for p in pairs), ";".join(p[1] for p in pairs)


# 产物表路线列及其缺省值（旧 parquet 缓存不含这些列，读取后需回填）
ROUTE_COLUMN_DEFAULTS: tuple[tuple[str, object], ...] = (
    ("route_ids", ""),
    ("route_redundancy", 1),
    ("route_steps_min", 1),
    ("green_score_best", 0.0),
)


def ensure_route_columns(df):
    """为旧产物表回填缺失的路线列（不改动已有列，返回同一个 DataFrame）。

    设计引擎在 Task 4/5 之前持久化的 parquet 不含 route_* 列；
    UI 读取这些缓存时不能抛 KeyError，故在此统一补齐默认值。
    """
    for col, default in ROUTE_COLUMN_DEFAULTS:
        if col not in df.columns:
            df[col] = default
    return df


def run_combinatorial_monomer_design(
    selected_precursor_ids: list[str] | None = None,
    custom_precursors: list[PrecursorCore] | None = None,
    selected_reaction_ids: list[str] | None = None,
    selected_rings: list[str] | None = None,
    selected_linkers: list[str] | None = None,
    selected_r_groups: list[str] | None = None,
    enable_scaffold_fission: bool = True,
    min_functionality: int = 2,
    max_sa_score: float = 6.0,
    mw_range: tuple[float, float] = (140.0, 1500.0),
    max_total_products: int = 5000,
    n_jobs: int = -1,
    enable_route_chains: bool = True,
    route_chain_ids: list[str] | None = None,
    progress_callback: Any | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Execute full two-stage combinatorial molecule design with multi-core parallel acceleration."""
    if not RDKIT_AVAILABLE:
        return pd.DataFrame(), ["RDKit is not available in the current Python environment."]

    if route_chain_ids is None:
        route_chain_ids = [r.route_id for r in ROUTE_CHAINS if r.default_enabled]

    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

    cpu_count = os.cpu_count() or 4
    # 在 512 核等超多核 Windows 服务器上，启用 128 个并发 worker，结合滑动窗口实现核心连续打满
    default_max_workers = min(128, cpu_count)
    workers = default_max_workers if (n_jobs is None or n_jobs < 1) else min(cpu_count, n_jobs)

    logs = [f"🚀 启动全维度虚拟分子设计流水线（已启用 {workers} 核心连续流式并行加速）..."]

    # 1. 准备前驱体母核库
    if custom_precursors:
        precursors = list(custom_precursors)
        if enable_scaffold_fission:
            fission_cores = generate_scaffold_intermediates(
                selected_rings=selected_rings,
                selected_linkers=selected_linkers,
                selected_r_groups=selected_r_groups,
                max_intermediates=max(500, min(100000, max_total_products // 2)),
            )
            precursors.extend(fission_cores)
            logs.append(f"🧬 阶段 1：融合自建母核 ({len(custom_precursors)} 个) 与骨架裂变库 ({len(fission_cores)} 个)，共计 {len(precursors)} 个前驱体")
        else:
            logs.append(f"📁 阶段 1：使用自建数据库提取的 {len(precursors)} 个前驱体母核")
    elif enable_scaffold_fission:
        precursors = generate_scaffold_intermediates(
            selected_rings=selected_rings,
            selected_linkers=selected_linkers,
            selected_r_groups=selected_r_groups,
            max_intermediates=max(500, min(100000, max_total_products // 2)),
        )
        logs.append(f"🧬 阶段 1：完成全维度拓扑骨架裂变组装，生成 {len(precursors)} 个多元前驱体母核")
    else:
        if not selected_precursor_ids:
            precursors = [p for p in PRECURSOR_CATALOG if p.default_selected]
        else:
            precursors = [p for p in PRECURSOR_CATALOG if p.core_id in selected_precursor_ids]
        logs.append(f"📋 阶段 1：使用经典前驱体母核库（共 {len(precursors)} 个母核）")

    # 2. 准备多通道反应模板
    if not selected_reaction_ids:
        reactions = [r for r in SYNTHETIC_REACTION_TEMPLATES if r.default_enabled]
    else:
        reactions = [r for r in SYNTHETIC_REACTION_TEMPLATES if r.reaction_id in selected_reaction_ids]

    logs.append(f"🧪 阶段 2：激活 {len(reactions)} 大多通道合成转化模板进行深度穷举装配")

    compiled_reactions = []
    for rxn_def in reactions:
        if not rxn_def.reaction_smarts:
            continue
        try:
            rxn = rdChemReactions.ReactionFromSmarts(rxn_def.reaction_smarts)
            compiled_reactions.append((rxn_def, rxn))
        except Exception as e:
            logs.append(f"⚠️ 反应模板 {rxn_def.name} 编译失败：{e}")

    # 预编译多步路线链 SMARTS 反应对象（避免每次进入循环反复解析耗时）
    compiled_routes = []
    if enable_route_chains:
        for route in ROUTE_CHAINS:
            if route_chain_ids is not None and route.route_id not in route_chain_ids:
                continue
            steps_rxns = []
            for step in route.steps:
                try:
                    steps_rxns.append(rdChemReactions.ReactionFromSmarts(step.reaction_smarts))
                except Exception:
                    steps_rxns.append(None)
            compiled_routes.append((route, steps_rxns))

    results: list[CombinatorialProduct] = []
    seen_idx: dict[str, int] = {}
    current_usable_count = 0

    from .task_manager import is_cancelled
    completed_precursors = 0
    total_precursors = len(precursors)

    # 批次粒度并发 + 平滑提前收敛：
    # 当用户设置了产物上限 max_total_products 时，如果已累计收集到足够多的可用单体
    # （按目标数 1.15 倍作为安全边际），立即停止投递和等待剩余数千个冗余前驱体，
    # 彻底解决“卡在阶段2达1小时”的恶性等待，数十秒即可高速收敛产出！
    target_usable_buffer = int(max_total_products * 1.15) if max_total_products > 0 else 0
    reached_target_early = False

    if workers > 1 and len(precursors) > 10:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            p_iter = iter(precursors)
            futures: dict[Any, PrecursorCore] = {}

            # 初始填满并发流式管道 (workers * 2 个任务，保持高吞吐且占用极低内存)
            initial_fill = min(workers * 2, total_precursors)
            for _ in range(initial_fill):
                p = next(p_iter, None)
                if p is None:
                    break
                fut = executor.submit(
                    _process_single_precursor_reactions,
                    p,
                    compiled_reactions,
                    reactions,
                    min_functionality,
                    max_sa_score,
                    mw_range,
                    enable_route_chains=enable_route_chains,
                    route_chain_ids=route_chain_ids,
                    compiled_routes=compiled_routes,
                )
                futures[fut] = p

            # 连续滑动窗口执行：任意一个完成立即填充下一个，杜绝批次屏障等待
            while futures:
                if is_cancelled():
                    logs.append("🛑 检测到用户后台终止指令，正在安全退出计算...")
                    for rem_f in futures:
                        rem_f.cancel()
                    break

                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    completed_precursors += 1
                    try:
                        p_res = fut.result()
                        for item in p_res:
                            if _merge_product_records(results, seen_idx, item):
                                current_usable_count += 1
                    except Exception:
                        pass
                    del futures[fut]

                    # 连续滑动填充：只要完成一个，立即无等待投递下一个母核
                    if not reached_target_early and not is_cancelled():
                        next_p = next(p_iter, None)
                        if next_p is not None:
                            new_fut = executor.submit(
                                _process_single_precursor_reactions,
                                next_p,
                                compiled_reactions,
                                reactions,
                                min_functionality,
                                max_sa_score,
                                mw_range,
                                enable_route_chains=enable_route_chains,
                                route_chain_ids=route_chain_ids,
                                compiled_routes=compiled_routes,
                            )
                            futures[new_fut] = next_p

                if progress_callback:
                    try:
                        progress_callback(
                            completed_precursors,
                            total_precursors,
                            current_usable_count,
                            f"已处理 {completed_precursors}/{total_precursors} 个母核，实时发现 {current_usable_count:,} 个可用单体..."
                        )
                    except Exception:
                        pass

                # 实时提前平滑收敛检查 (O(1) 无需线性扫描)
                if target_usable_buffer > 0 and current_usable_count >= target_usable_buffer and not reached_target_early:
                    reached_target_early = True
                    logs.append(f"✨ 目标配额达成：已累计提取到 {current_usable_count:,} 个高品质单体（已达预设上限 {max_total_products:,} 个），触发平滑提前收敛")
                    for remaining_fut in futures:
                        remaining_fut.cancel()
                    break
    else:
        # 单核降级执行
        for precursor in precursors:
            if is_cancelled() or reached_target_early:
                break
            completed_precursors += 1
            p_res = _process_single_precursor_reactions(
                precursor,
                compiled_reactions,
                reactions,
                min_functionality,
                max_sa_score,
                mw_range,
                enable_route_chains=enable_route_chains,
                route_chain_ids=route_chain_ids,
                compiled_routes=compiled_routes,
            )
            for item in p_res:
                if is_cancelled():
                    break
                if _merge_product_records(results, seen_idx, item):
                    current_usable_count += 1

            if progress_callback:
                try:
                    progress_callback(
                        completed_precursors,
                        total_precursors,
                        current_usable_count,
                        f"已处理 {completed_precursors}/{total_precursors} 个母核，实时发现 {current_usable_count:,} 个可用单体..."
                    )
                except Exception:
                    pass

            if target_usable_buffer > 0 and current_usable_count >= target_usable_buffer:
                reached_target_early = True
                logs.append(f"✨ 目标配额达成：已累计提取到 {current_usable_count:,} 个高品质单体（已达预设上限 {max_total_products:,} 个），触发平滑提前收敛")
                break

    # 确定性排序后再截断：相同输入参数必然得到相同产物集合
    results.sort(key=lambda r: (r.product_smiles, r.reaction_id, r.precursor_id))
    # 截断只作用于可用单体（resin/hardener）：max_total_products 是用户对"单体产物数"的预期，
    # 中间体（role=intermediate）是路线链的副产物、不进配方筛选，不应挤占该名额。
    # 否则开启路线链后可用单体会被中间体大幅吃掉（实测 3000 → 1252，-58%）。
    usable = [r for r in results if r.role != "intermediate"]
    intermediates_only = [r for r in results if r.role == "intermediate"]
    if len(usable) > max_total_products:
        usable = usable[:max_total_products]
    results = usable + intermediates_only

    logs.append(f"🎉 阶段 3：质量门禁校验完成！成功产出 {len(usable):,} 个高性能单体"
                + (f"（另有 {len(intermediates_only):,} 个路线中间体，不参与配方筛选）" if intermediates_only else ""))

    if not results:
        return pd.DataFrame(), logs

    df = pd.DataFrame([asdict(r) for r in results])
    if not df.empty:
        # 路线 id/名称成对按 route_id 排序后再序列化：与线程完成顺序无关，
        # 保证 route_ids/route_names 跨运行字节级一致（排序逻辑见 _serialize_route_columns）
        serialized = [_serialize_route_columns(r.route_ids, r.route_names) for r in results]
        df["route_ids"] = [s[0] for s in serialized]
        df["route_names"] = [s[1] for s in serialized]
        # [规范去重] 产物入库前基于 RDKit Canonical SMILES 做化学等价严格去重，
        # 杜绝同分异构/不同书写形式在产物库中产生重复分子。
        df = deduplicate_monomers_canonical(df, smiles_col="product_smiles")
    return df, logs


def deduplicate_monomers_canonical(
    df: pd.DataFrame,
    smiles_col: str = "product_smiles",
    *,
    keep: str = "first",
) -> pd.DataFrame:
    """基于 RDKit Canonical SMILES 对单体候选库执行化学等价严格去重。

    解决用户痛点：「候选池最好不要有重复的分子」。
    单纯依靠字符串 drop_duplicates 会放过书写形式不同但化学等价的同一分子；
    本函数将每行转换为标准规范 SMILES 后进行唯一性归并；
    若无法通过 RDKit 解析则退化为清洗后的原始文本去重，绝不静默丢弃候选。
    """
    if not isinstance(df, pd.DataFrame) or df.empty or smiles_col not in df.columns:
        return df

    try:
        from .smiles_utils import canonicalize_smiles
    except ImportError:
        try:
            from smiles_utils import canonicalize_smiles
        except ImportError:
            canonicalize_smiles = None

    unique_raw = df[smiles_col].dropna().unique()
    canon_map = {}
    for s in unique_raw:
        s_clean = str(s).strip()
        if not s_clean:
            canon_map[s] = ""
            continue
        can = canonicalize_smiles(s_clean) if canonicalize_smiles is not None else None
        canon_map[s] = can if can is not None else s_clean

    work = df.copy()
    work["_canonical_key"] = work[smiles_col].map(canon_map).fillna(work[smiles_col].astype(str))
    work = work[work["_canonical_key"].str.strip().ne("")]
    deduped = work.drop_duplicates(subset=["_canonical_key"], keep=keep)
    # 将标准化的 canonical smiles 写回原 smiles 列，保证库内所有分子表示绝对统一规范
    deduped = deduped.copy()
    deduped[smiles_col] = deduped["_canonical_key"]
    return deduped.drop(columns=["_canonical_key"]).reset_index(drop=True)

