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
        reaction_smarts="[#6:1]-[OX2H:2]>>[#6:1]-OCC1CO1",
        description="多元酚/脂肪多元醇 + 环氧氯丙烷生成缩水甘油醚环氧树脂 (DGEBA/氢化双酚A/脂环多元醇醚型)",
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
        reaction_smarts="[C:1]=[C:2]>>[C:1]1O[C:2]1",
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

def generate_scaffold_intermediates(
    selected_rings: list[str] | None = None,
    selected_linkers: list[str] | None = None,
    selected_r_groups: list[str] | None = None,
    include_phenols: bool = True,
    include_amines: bool = True,
    include_acids: bool = True,
    max_intermediates: int = 10000,
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
    ]

    for r_name, r_desc, ring_key, ring_base, p_tmpl, a_tmpl in ring_templates:
        if not _ring_enabled(ring_key):
            continue
        for linker in linkers:
            if linker.valency != 2:
                continue
            for r_grp in r_groups:
                if len(intermediates) >= max_intermediates:
                    break

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
    ]
    for c_id, c_name, ring_key, p_tmpl, a_tmpl in poly_cores:
        if not _ring_enabled(ring_key):
            continue
        for r_grp in r_groups:
            if len(intermediates) >= max_intermediates:
                break
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

    # 2. 三核星型组装模型 (>CH- 次甲基核，由 methine_star 桥联库控制)
    if _linker_enabled("methine_star") and _ring_enabled("benzene"):
        for r_grp in r_groups:
            if len(intermediates) >= max_intermediates:
                break
            r_str = f"({r_grp.smiles})" if r_grp.smiles else ""
            smi_star_p = f"Oc1cc{r_str}c(C(c2ccc(O)cc2)c2ccc(O)cc2)cc1"
            smi_star_a = f"Nc1cc{r_str}c(C(c2ccc(N)cc2)c2ccc(N)cc2)cc1"
            for s, cat, role, prefix in [(smi_star_p, "拓扑星型多酚", "resin", "三酚"), (smi_star_a, "拓扑星型多胺", "hardener", "三胺")]:
                try:
                    m = Chem.MolFromSmiles(s)
                    if m:
                        Chem.SanitizeMol(m)
                        cs = Chem.MolToSmiles(m, canonical=True)
                        if cs not in seen_smiles:
                            seen_smiles.add(cs)
                            intermediates.append(
                                PrecursorCore(
                                    core_id=f"gen_star_{hashlib.md5(cs.encode()).hexdigest()[:8]}",
                                    name=f"星型{prefix} ({r_grp.name})",
                                    category=cat,
                                    role=role,
                                    smiles=cs,
                                    description=f"三官能度星型拓扑核心前驱体",
                                    default_selected=True,
                                )
                            )
                except Exception:
                    pass

    # 3. 空间四官能组装模型 (四苯乙烷/四苯甲烷/卡基芴/螺双二氢茚，由对应环系与星型桥联控制)
    for r_grp in r_groups:
        if len(intermediates) >= max_intermediates:
            break
        r_str = f"({r_grp.smiles})" if r_grp.smiles else ""
        spatial_candidates = []
        if _linker_enabled("ethane_tetra") and _ring_enabled("benzene"):
            spatial_candidates.extend([
                (f"Oc1cc{r_str}c(C(c2ccc(O)cc2)C(c2ccc(O)cc2)c2ccc(O)cc2)cc1", "四苯乙烷四酚", "resin"),
                (f"Nc1cc{r_str}c(C(c2ccc(N)cc2)C(c2ccc(N)cc2)c2ccc(N)cc2)cc1", "四苯乙烷四胺", "hardener"),
            ])
        if _linker_enabled("quaternary_carbon") and _ring_enabled("benzene"):
            spatial_candidates.extend([
                (f"Oc1cc{r_str}c(C(c2ccc(O)cc2)(c2ccc(O)cc2)c2ccc(O)cc2)cc1", "四苯甲烷四酚", "resin"),
                (f"Nc1cc{r_str}c(C(c2ccc(N)cc2)(c2ccc(N)cc2)c2ccc(N)cc2)cc1", "四苯甲烷四胺", "hardener"),
            ])
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
                if len(intermediates) >= max_intermediates:
                    break
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
    if include_acids:
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
                if len(intermediates) >= max_intermediates:
                    break
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

    for retro_id, smarts, desc in RETRO_DECONSTRUCTION_TEMPLATES:
        try:
            rxn = rdChemReactions.ReactionFromSmarts(smarts)
        except Exception:
            continue

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

                # 按官能团判别角色（不能用 N/O 元素有无判断：醚键 O 会让 ODA/Jeffamine
                # 等二胺被误判为树脂）
                patt_nh = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")
                patt_phenol = Chem.MolFromSmarts("[cX3][OX2H]")
                patt_acid = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
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
        current_mols = next_mols

    return current_mols if reacted_ever else []


def _precursor_matches_reaction(parent_mol, rxn) -> bool:
    """用反应物侧 SMARTS 模板实测底物是否真的能反应，替代按类别字符串猜测的预过滤。"""
    try:
        for reactant_template in rxn.GetReactants():
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
    )


def _process_single_precursor_reactions(
    precursor: PrecursorCore,
    compiled_reactions: list[tuple[SyntheticReaction, Any]],
    reactions: list[SyntheticReaction],
    min_functionality: int,
    max_sa_score: float,
    mw_range: tuple[float, float],
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
    for rxn_def, rxn in compiled_reactions:
        if is_cancelled():
            break

        # 预过滤：用反应物侧 SMARTS 实测底物能否反应（取代按类别字符串猜测，
        # 避免脂肪胺被芳香模板放行后原样穿透成“假树脂”）
        if not _precursor_matches_reaction(parent_mol, rxn):
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

    return local_results


# ==============================================================================
# 6. 全流水线正交展开与配方级高通量衔接引擎 (多核并行加速)
# ==============================================================================

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
) -> tuple[pd.DataFrame, list[str]]:
    """Execute full two-stage combinatorial molecule design with multi-core parallel acceleration."""
    if not RDKIT_AVAILABLE:
        return pd.DataFrame(), ["RDKit is not available in the current Python environment."]

    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cpu_count = os.cpu_count() or 4
    workers = cpu_count if (n_jobs is None or n_jobs < 1) else min(cpu_count, n_jobs)

    logs = [f"🚀 启动全维度虚拟分子设计流水线（已启用 {workers} 核心多线程并行加速）..."]

    # 1. 准备前驱体母核库
    if custom_precursors:
        precursors = list(custom_precursors)
        if enable_scaffold_fission:
            fission_cores = generate_scaffold_intermediates(
                selected_rings=selected_rings,
                selected_linkers=selected_linkers,
                selected_r_groups=selected_r_groups,
                max_intermediates=max(500, max_total_products // 2),
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
            max_intermediates=max(500, max_total_products // 2),
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

    results: list[CombinatorialProduct] = []
    seen_smiles = set()

    # 3. 多核并发正交展开
    # 注意：为保证结果绝对可复现，并发模式下先完整收集全部产物，
    # 统一按 canonical SMILES 排序后再截断；不允许按线程完成顺序"先到先得"。
    from .task_manager import is_cancelled
    if workers > 1 and len(precursors) > 10:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _process_single_precursor_reactions,
                    p,
                    compiled_reactions,
                    reactions,
                    min_functionality,
                    max_sa_score,
                    mw_range,
                )
                for p in precursors
            ]
            for fut in as_completed(futures):
                if is_cancelled():
                    logs.append("🛑 检测到用户后台终止指令，正在安全退出计算...")
                    break
                try:
                    p_res = fut.result()
                    for item in p_res:
                        if item.product_smiles not in seen_smiles:
                            seen_smiles.add(item.product_smiles)
                            results.append(item)
                except Exception:
                    pass
    else:
        # 单核降级执行
        for precursor in precursors:
            if is_cancelled():
                logs.append("🛑 检测到用户后台终止指令，正在安全退出计算...")
                break
            p_res = _process_single_precursor_reactions(
                precursor,
                compiled_reactions,
                reactions,
                min_functionality,
                max_sa_score,
                mw_range,
            )
            for item in p_res:
                if is_cancelled():
                    break
                if item.product_smiles not in seen_smiles:
                    seen_smiles.add(item.product_smiles)
                    results.append(item)

    # 确定性排序后再截断：相同输入参数必然得到相同产物集合
    results.sort(key=lambda r: (r.product_smiles, r.reaction_id, r.precursor_id))
    if len(results) > max_total_products:
        results = results[:max_total_products]

    logs.append(f"🎉 阶段 3：质量门禁校验完成！成功产出 {len(results):,} 个高性能单体")

    if not results:
        return pd.DataFrame(), logs

    df = pd.DataFrame([asdict(r) for r in results])
    return df, logs
