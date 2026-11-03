"""
重构特征提取器测试套件 (Test Rigorous Extractors)
验证核心：
1. 拓扑提取器：从真实单体与反应加成键提取转化率，未严格解耦的弹性链段置为 null 并标为 NOT_VERIFIED；
2. 自由体积提取器：周期性边界条件 (PBC) 刚体平移不变性，未做渗流绕行连通性的孔隙体积分数置为 null；
3. 相互作用提取器：单位换算正确性 (KJ_MOL_NM3_TO_MJ_M3)，保留物理正负号，未定义解聚气相孤立参考态的 CED 严格置为 null；基于共价键拓扑的氢键识别。
"""

import math
import numpy as np
import pytest

from src.feature_extractors import TopologyExtractor, FreeVolumeExtractor, InteractionExtractor

def test_rigorous_topology_extractor():
    extractor = TopologyExtractor()

    # 构造单体列表：2 个 DGEBA (49原子, mw=340.4, 2官能), 1 个 4,4-DDS (29原子, mw=248.3, 4官能)
    monomers = [
        {"index": 0, "name": "DGEBA_1", "type": "epoxy", "mw": 340.4},
        {"index": 1, "name": "DGEBA_2", "type": "epoxy", "mw": 340.4},
        {"index": 2, "name": "DDS_1", "type": "hardener", "mw": 248.3}
    ]

    # 实际形成的反应交联键：DGEBA_1 与 DDS_1 成键，DGEBA_2 与 DDS_1 成键
    crosslink_bonds = [(0, 2), (1, 2)]
    # 初始 4 个环氧基
    res = extractor.extract_from_covalent_network(
        monomer_units=monomers,
        reaction_crosslink_bonds=crosslink_bonds,
        initial_epoxy_count=4,
        box_volume_nm3=8.0
    )

    assert res["md_alpha_actual"] == 0.50  # 2 / 4 = 0.50
    assert res["md_largest_cluster_fraction"] == 1.0  # 全网络连通
    assert res["md_strand_mw_mean_g_mol"] is None  # 严格置为 null
    assert res["md_strand_verification_status"] == "NOT_VERIFIED"
    assert res["qc_actual_crosslink_bonds"] == 2

def test_rigorous_free_volume_extractor(tmp_path):
    extractor = FreeVolumeExtractor(grid_resolution_nm=0.08)

    box_lengths = np.array([2.5, 2.5, 2.5])
    coords = np.array([
        [0.5, 0.5, 0.5],
        [1.2, 1.2, 1.2],
        [1.8, 0.8, 1.5],
        [0.3, 2.1, 0.9]
    ])
    elements = ["C", "O", "N", "H"]

    res1 = extractor.extract_free_volume_grid(coords, elements, box_lengths, probe_radii_nm=[0.0, 0.12])
    assert "md_ffv_probe_0A" in res1
    assert "md_ffv_probe_1_2A" in res1
    assert res1["md_pore_volume_fraction"] is None  # 未做 3D 渗流绕行连通性，严格置为 null
    assert res1["md_pore_verification_status"] == "NOT_VERIFIED"

    # 刚体平移与周期性包裹不变性测试
    shift = np.array([0.7, -1.1, 1.9])
    shifted_coords = np.mod(coords + shift, box_lengths)
    res2 = extractor.extract_free_volume_grid(shifted_coords, elements, box_lengths, probe_radii_nm=[0.0, 0.12])

    assert abs(res1["md_density_g_cm3"] - res2["md_density_g_cm3"]) < 1e-4
    assert abs(res1["md_ffv_probe_0A"] - res2["md_ffv_probe_0A"]) < 0.015
    assert abs(res1["md_ffv_probe_1_2A"] - res2["md_ffv_probe_1_2A"]) < 0.015

    # 未知元素报错测试
    with pytest.raises(ValueError, match="未知元素"):
        extractor.get_vdw_radius("UnknownElem")

def test_rigorous_interaction_extractor():
    extractor = InteractionExtractor()

    # 1. 能量密度换算与 null-CED 测试
    # -500 kJ/mol 在 10 nm3 盒子中
    ed_res = extractor.calculate_energy_densities(
        vdw_energy_kJ_mol=-500.0,
        elec_energy_kJ_mol=-1200.0,
        box_volume_nm3=10.0
    )
    expected_vdw = (-500.0 / 10.0) * 1.66053906660
    assert abs(ed_res["md_vdw_energy_density_MJ_m3"] - expected_vdw) < 0.5
    assert ed_res["md_vdw_energy_density_MJ_m3"] < 0.0  # 保留物理负号 (吸引势能)
    assert ed_res["md_ced_MJ_m3"] is None  # 彻底删除伪 CED，严格输出 null
    assert ed_res["md_ced_verification_status"] == "NOT_VERIFIED"

    # 2. 基于真实共价键的氢键识别测试
    coords = np.array([
        [1.0, 1.0, 1.0],     # 0: O (donor)
        [1.0, 1.0, 1.096],   # 1: H
        [1.0, 1.0, 1.280],   # 2: N (acceptor)
        [0.2, 0.2, 0.2]      # 3: C (non-polar)
    ])
    elements = ["O", "H", "N", "C"]
    box = np.array([3.0, 3.0, 3.0])
    # 明确定义真实共价键：0号O 与 1号H 形成共价 O-H 键
    covalent_bonds = [(0, 1)]

    hb_res = extractor.extract_hydrogen_bonds_pbc(coords, elements, covalent_bonds, box)
    assert hb_res["qc_hbond_count_total"] == 1
    assert hb_res["md_hbond_density_nm3"] > 0.0
