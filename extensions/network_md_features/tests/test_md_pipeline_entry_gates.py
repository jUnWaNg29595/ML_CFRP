"""
test_md_pipeline_entry_gates.py
测试核心科学门禁、PBC 接触与共价键长检查、电荷协议核验及新后处理交付物的一致性。
"""

import os
import sys
import pytest
import os
import sys
import shutil
import pytest
import numpy as np
import pandas as pd
import parmed as pmd
from pathlib import Path

EXT_ROOT = Path("/home/wangj/project/CFRP/extensions/network_md_features")
if str(EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXT_ROOT))

ATTEMPT_DIR = EXT_ROOT / "artifacts" / "attempt_20261019_01"
POSTPROCESS_DIR = EXT_ROOT / "artifacts" / "postprocess_verified_20261019_02"

from src.assembly_engine import (
    check_inter_residue_contacts,
    check_system_bond_strains,
    inject_and_verify_resp_mol2,
)
from src.postprocess_trajectory import audit_state_charge_protocol
from src.compute_features_from_trajectory import recompute_features_for_attempt


def test_adversarial_1_all_zero_charges_with_zero_net_rejected():
    """全零单体电荷即使净电荷为 0，调用真实生产函数 audit_state_charge_protocol 也必须抛出 ValueError"""
    prmtop_p = ATTEMPT_DIR / "state_0_addition" / "system.prmtop"
    parm = pmd.load_file(str(prmtop_p))
    for a in parm.atoms:
        a.charge = 0.0

    with pytest.raises(ValueError, match="全部原子电荷为 0"):
        audit_state_charge_protocol(parm)


def test_adversarial_2_swapped_resp_charge_mapping_detected():
    """单体电荷顺序错位时，audit_state_charge_protocol 必须识别并拒绝赋予 GAFF2_MONOMER_DFT_RESP_ORDERED_MATCH"""
    prmtop_p = ATTEMPT_DIR / "state_0_addition" / "system.prmtop"
    parm = pmd.load_file(str(prmtop_p))
    # 交换前两个原子的电荷
    q0, q1 = parm.atoms[0].charge, parm.atoms[1].charge
    parm.atoms[0].charge = q1
    parm.atoms[1].charge = q0

    res = audit_state_charge_protocol(parm)
    assert res["charge_protocol"] != "GAFF2_MONOMER_DFT_RESP_ORDERED_MATCH"
    assert res["validation_status"] == "DIAGNOSTIC_UNVERIFIED_CHARGE"


def test_adversarial_3_severe_contact_clash_including_pbc_boundary():
    """跨残基碰撞（包括跨越周期边界的 0.2 A 碰撞）必须被 check_inter_residue_contacts 检出"""
    prmtop_p = ATTEMPT_DIR / "state_0_addition" / "system.prmtop"
    gro_p = ATTEMPT_DIR / "state_0_addition" / "system.gro"
    parm = pmd.load_file(str(prmtop_p), xyz=str(gro_p))

    coords = parm.coordinates.copy()
    box_x = float(parm.box[0])  # 55.0 A
    r1_first = [a.idx for a in parm.atoms if a.residue.idx == 0][0]
    r2_first = [a.idx for a in parm.atoms if a.residue.idx == 1][0]
    coords[r1_first] = np.array([0.1, 10.0, 10.0])
    coords[r2_first] = np.array([box_x - 0.1, 10.0, 10.0])  # 跨周期边界距离 0.2 A
    parm.coordinates = coords

    min_dist, clash_pair = check_inter_residue_contacts(parm, min_allowed_dist_A=3.0)
    assert abs(min_dist - 0.2) < 1e-5
    assert min_dist < 3.0


def test_adversarial_4_stretched_bond_despite_converged_force():
    """共价键拉伸至 2.47 A 时，check_system_bond_strains 必须捕获异常键"""
    prmtop_p = ATTEMPT_DIR / "state_0_addition" / "system.prmtop"
    gro_p = ATTEMPT_DIR / "state_0_addition" / "min.gro"
    parm = pmd.load_file(str(prmtop_p), xyz=str(gro_p))

    coords = parm.coordinates.copy()
    b = parm.bonds[0]
    a1, a2 = b.atom1.idx, b.atom2.idx
    coords[a2] = coords[a1] + np.array([2.47, 0.0, 0.0])

    max_dev, abnormal_bonds = check_system_bond_strains(parm, coords=coords, max_allowed_dev_A=0.25)
    assert max_dev > 0.9
    assert len(abnormal_bonds) >= 1


def test_adversarial_5_lifecycle_identity_and_unverified_cross_state():
    """核验生命周期表与 State 0 真实拓扑元素 100% 一致，且未证明跨态双射的 State 1/2 状态标记为 NOT_VERIFIED"""
    prmtop_p = ATTEMPT_DIR / "state_0_addition" / "system.prmtop"
    parm = pmd.load_file(str(prmtop_p))

    csv_p = POSTPROCESS_DIR / "atom_lifecycle_state0_verified.csv"
    assert csv_p.exists()
    df_life = pd.read_csv(csv_p)
    assert len(df_life) == 254

    for idx, row in df_life.iterrows():
        assert parm.atoms[idx].element_name == row["element"]

    assert (df_life["state_1_status"] == "NOT_VERIFIED_INDEPENDENT_PRESET_TOPOLOGY").all()
    assert (df_life["state_2_status"] == "NOT_VERIFIED_INDEPENDENT_PRESET_TOPOLOGY").all()
    assert (df_life["cross_state_bijection_verified"] == False).all()


def test_adversarial_6_missing_attempt_or_missing_xtc_edr_rejected(tmp_path):
    """不存在的目录或仅有 prmtop+gro 而无 xtc/edr 的目录必须抛出 FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        recompute_features_for_attempt(str(tmp_path / "non_existent_attempt"))

    without_traj = tmp_path / "without_xtc_edr"
    st0 = without_traj / "state_0_addition"
    st0.mkdir(parents=True)
    shutil.copy2(ATTEMPT_DIR / "state_0_addition" / "nvt.gro", st0 / "nvt.gro")
    shutil.copy2(ATTEMPT_DIR / "state_0_addition" / "system.prmtop", st0 / "system.prmtop")
    shutil.copy2(ATTEMPT_DIR / "state_0_addition" / "nvt.gro", st0 / "nvt.gro")

    with pytest.raises(FileNotFoundError):
        recompute_features_for_attempt(str(without_traj), str(tmp_path / "out_fail"))


def test_verified_postprocess_20261019_02_matches_native_audit_baselines():
    """核验新独立目录 postprocess_verified_20261019_02 的真实能量、几何与诚实标签"""
    feat_csv = POSTPROCESS_DIR / "md_features_diagnostic.csv"
    assert feat_csv.exists()
    df_f = pd.read_csv(feat_csv)
    assert len(df_f) == 3

    # 1. 诚实协议与状态标签
    assert df_f.loc[0, "validation_status"] == "DIAGNOSTIC_SMALL_SPARSE_RESP_REFERENCE"
    assert df_f.loc[0, "charge_protocol"] == "GAFF2_MONOMER_DFT_RESP_ORDERED_MATCH"
    assert df_f.loc[1, "validation_status"] == "DIAGNOSTIC_SMALL_SPARSE_MIXED_CHARGE"
    assert df_f.loc[1, "charge_protocol"] == "GAFF2_UNREACTED_RESP_ADDUCT_GASTEIGER_MIXED_DIAGNOSTIC"
    assert df_f.loc[2, "charge_protocol"] == "GAFF2_UNREACTED_RESP_ADDUCT_GASTEIGER_MIXED_DIAGNOSTIC"
    assert (df_f["admitted_to_formal_ml_training"] == 0).all()

    # 2. 原生 EDR 能量密度基线核对 (t=0~10 ps, 101 帧算术平均)
    assert abs(df_f.loc[0, "md_vdw_energy_density_MJ_m3"] - 1.79289756) < 1e-5
    assert abs(df_f.loc[0, "md_coulomb_energy_density_MJ_m3"] - 3.38289119) < 1e-5
    assert abs(df_f.loc[0, "sampling_temperature_K"] - 300.211170) < 1e-4
    assert abs(df_f.loc[0, "md_potential_energy_mean_kJ_mol"] - 1847.081294) < 1e-4

    assert abs(df_f.loc[1, "md_vdw_energy_density_MJ_m3"] - 1.49325204) < 1e-5
    assert abs(df_f.loc[1, "md_coulomb_energy_density_MJ_m3"] - 0.53053784) < 1e-5

    assert abs(df_f.loc[2, "md_vdw_energy_density_MJ_m3"] - 0.76708998) < 1e-5
    assert abs(df_f.loc[2, "md_coulomb_energy_density_MJ_m3"] - 0.24647022) < 1e-5

    # 3. 严格 CED 保持 null
    assert df_f["md_cohesive_energy_density_CED_MPa"].isna().all()
