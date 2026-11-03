"""
交联网络物理特征数据模式 (Feature Schema)
定义所有 md_* 物理特征的命名、单位、物理意义、门禁要求和置信状态。
严格遵循科学规范：绝不将代理值、未平衡值或未经严格热力学定义的势能冒充为真值。
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional

@dataclass
class FeatureDefinition:
    name: str
    unit: str
    dtype: str
    description: str
    physical_category: str  # "topology", "free_volume", "interaction", "quality_control"
    verification_level: str  # "VERIFIED", "EXPLORATORY", "DIAGNOSTIC_ONLY", "NOT_VERIFIED"
    min_reasonable_val: Optional[float] = None
    max_reasonable_val: Optional[float] = None
    requires_gate: str = "S7"

MD_FEATURE_REGISTRY: Dict[str, FeatureDefinition] = {
    # 1. 网络拓扑特征 (Network Topology)
    "md_alpha_actual": FeatureDefinition(
        name="md_alpha_actual",
        unit="dimensionless",
        dtype="float64",
        description="实际有效环氧加成数与初始环氧基团总数的比值（依据最终真实共价成键图计算，非预设目标值）",
        physical_category="topology",
        verification_level="VERIFIED",
        min_reasonable_val=0.0,
        max_reasonable_val=1.0,
        requires_gate="S5"
    ),
    "md_largest_cluster_fraction": FeatureDefinition(
        name="md_largest_cluster_fraction",
        unit="dimensionless",
        dtype="float64",
        description="交联体系中最大连通团簇（渗流网络）所占全部原子质量的分数",
        physical_category="topology",
        verification_level="VERIFIED",
        min_reasonable_val=0.0,
        max_reasonable_val=1.0,
        requires_gate="S5"
    ),
    "md_junction_density_mol_cm3": FeatureDefinition(
        name="md_junction_density_mol_cm3",
        unit="mol/cm^3",
        dtype="float64",
        description="体系内有效交联节点（连接度 >= 3 的分支单体/官能点）的摩尔体积密度",
        physical_category="topology",
        verification_level="VERIFIED",
        min_reasonable_val=0.0,
        max_reasonable_val=0.05,
        requires_gate="S5"
    ),
    "md_strand_mw_mean_g_mol": FeatureDefinition(
        name="md_strand_mw_mean_g_mol",
        unit="g/mol",
        dtype="float64",
        description="有效交联点之间网络链段（Strands）的数均分子量（排除悬垂链与小分子孤立单体）",
        physical_category="topology",
        verification_level="VERIFIED",
        min_reasonable_val=50.0,
        max_reasonable_val=5000.0,
        requires_gate="S5"
    ),
    "md_dangling_chain_fraction": FeatureDefinition(
        name="md_dangling_chain_fraction",
        unit="dimensionless",
        dtype="float64",
        description="仅一端与主网络相连的悬垂链段（Dangling Chains）占全部链段的摩尔比例",
        physical_category="topology",
        verification_level="VERIFIED",
        min_reasonable_val=0.0,
        max_reasonable_val=1.0,
        requires_gate="S5"
    ),
    "md_cycle_rank": FeatureDefinition(
        name="md_cycle_rank",
        unit="dimensionless",
        dtype="int64",
        description="网络拓扑图的环秩（独立闭合回路数，E - V + C）",
        physical_category="topology",
        verification_level="VERIFIED",
        min_reasonable_val=0.0,
        max_reasonable_val=10000.0,
        requires_gate="S5"
    ),

    # 2. 堆积与自由体积特征 (Packing & Free Volume)
    "md_density_g_cm3": FeatureDefinition(
        name="md_density_g_cm3",
        unit="g/cm^3",
        dtype="float64",
        description="平衡态参考温度与压力下 MD 盒子的真实质量密度（非预设 1.2 g/cm3 静态假定）",
        physical_category="free_volume",
        verification_level="VERIFIED",
        min_reasonable_val=0.8,
        max_reasonable_val=1.6,
        requires_gate="S6"
    ),
    "md_ffv_probe_0A": FeatureDefinition(
        name="md_ffv_probe_0A",
        unit="dimensionless",
        dtype="float64",
        description="基于 Bondi 范德华半径的真实周期盒子几何未占据体积分数 (1 - V_vdw / V_box)",
        physical_category="free_volume",
        verification_level="VERIFIED",
        min_reasonable_val=0.10,
        max_reasonable_val=0.60,
        requires_gate="S7"
    ),
    "md_ffv_probe_1_2A": FeatureDefinition(
        name="md_ffv_probe_1_2A",
        unit="dimensionless",
        dtype="float64",
        description="采用 1.2 A 探针（类似 He/H2 分子）在 PBC 下探测的可达自由体积分数",
        physical_category="free_volume",
        verification_level="VERIFIED",
        min_reasonable_val=0.00,
        max_reasonable_val=0.35,
        requires_gate="S7"
    ),
    "md_pore_volume_fraction": FeatureDefinition(
        name="md_pore_volume_fraction",
        unit="dimensionless",
        dtype="float64",
        description="连通空穴（可渗流微孔通道）占总盒体积的分数",
        physical_category="free_volume",
        verification_level="VERIFIED",
        min_reasonable_val=0.00,
        max_reasonable_val=0.25,
        requires_gate="S7"
    ),

    # 3. 相互作用与能量密度特征 (Interactions & Energy Densities)
    "md_vdw_energy_density_MJ_m3": FeatureDefinition(
        name="md_vdw_energy_density_MJ_m3",
        unit="MJ/m^3",
        dtype="float64",
        description="体系范德华（Lennard-Jones）相互作用势能密度（声明力场下的分项能量/平衡体积）",
        physical_category="interaction",
        verification_level="VERIFIED",
        min_reasonable_val=-1500.0,
        max_reasonable_val=0.0,
        requires_gate="S7"
    ),
    "md_electrostatic_energy_density_MJ_m3": FeatureDefinition(
        name="md_electrostatic_energy_density_MJ_m3",
        unit="MJ/m^3",
        dtype="float64",
        description="体系静电库仑相互作用势能密度（PME/Reaction-Field 全长程静电/平衡体积）",
        physical_category="interaction",
        verification_level="VERIFIED",
        min_reasonable_val=-3000.0,
        max_reasonable_val=100.0,
        requires_gate="S7"
    ),
    "md_hbond_density_nm3": FeatureDefinition(
        name="md_hbond_density_nm3",
        unit="nm^-3",
        dtype="float64",
        description="根据几何判据（供体-受体距离 <= 0.35 nm 且 供体-氢-受体夹角 >= 120 度）统计的单位体积氢键数",
        physical_category="interaction",
        verification_level="VERIFIED",
        min_reasonable_val=0.0,
        max_reasonable_val=15.0,
        requires_gate="S7"
    ),
    "md_ced_MJ_m3": FeatureDefinition(
        name="md_ced_MJ_m3",
        unit="MJ/m^3",
        dtype="float64",
        description="内聚能密度（Cohesive Energy Density）。注：对共价交联全网络，若未严格隔离单体断裂或片段剥离参考态，标记为 NOT_VERIFIED",
        physical_category="interaction",
        verification_level="NOT_VERIFIED",
        min_reasonable_val=200.0,
        max_reasonable_val=800.0,
        requires_gate="S7"
    ),

    # 4. 模拟质量控制特征 (Quality Control & Provenance)
    "qc_density_drift_pct_per_ns": FeatureDefinition(
        name="qc_density_drift_pct_per_ns",
        unit="%/ns",
        dtype="float64",
        description="平衡采样段密度相对时间线性拟合漂移斜率，用以衡量参考态稳定度",
        physical_category="quality_control",
        verification_level="VERIFIED",
        min_reasonable_val=-10.0,
        max_reasonable_val=10.0,
        requires_gate="S6"
    ),
    "qc_box_size_nm": FeatureDefinition(
        name="qc_box_size_nm",
        unit="nm",
        dtype="float64",
        description="平衡态立方盒子的平均边长",
        physical_category="quality_control",
        verification_level="VERIFIED",
        min_reasonable_val=1.5,
        max_reasonable_val=25.0,
        requires_gate="S6"
    ),
    "qc_gate_status": FeatureDefinition(
        name="qc_gate_status",
        unit="string",
        dtype="object",
        description="S0~S8 科学门禁的综合通过状态 (PASS, FAIL, EXPLORATORY)",
        physical_category="quality_control",
        verification_level="VERIFIED",
        requires_gate="S8"
    )
}

def export_schema_json(output_path: str) -> None:
    schema_dict = {
        "title": "Crosslink Network Physical Features Schema",
        "version": "1.0.0",
        "description": "CFRP 独立交联网络物理特征数据模式规范",
        "features": {k: asdict(v) for k, v in MD_FEATURE_REGISTRY.items()}
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema_dict, f, indent=2, ensure_ascii=False)
    print(f"Exported schema to {output_path}")

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "/home/wangj/project/CFRP/extensions/network_md_features/md_feature_schema.json"
    export_schema_json(out)
