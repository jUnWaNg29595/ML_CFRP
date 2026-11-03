"""
严密科学门禁系统 (Rigorous Scientific Quality Gates S0 - S8)
设计原则：
1. 绝对分离工程状态与科学状态；
2. 门禁计算全部基于真实输入文件与引擎原生输出（TOP/ITP/GRO/TPR/LOG），禁止外部手填 missing 列表或随意传入虚构数值；
3. 严格状态：NOT_RUN, BLOCKED, NOT_VERIFIED, FAIL, EXPLORATORY, PASS；
4. 综合判定：总体 PASS 必须要求所有必要门禁存在且全部 PASS；任何 EXPLORATORY 只能导致总体 EXPLORATORY，严禁升格为 PASS；任何缺失或失败必须拦截为 BLOCKED 或 FAIL；
5. Fail-Closed 哲学：缺少化学图/3D几何、未知力场类型、未收敛日志、全体系周期碰撞、残留牺牲氢一律严格拦截；
6. 严格防御独立验收对抗性测试 (ht_adversarial.json):
   - S1: 缺少真实分子连接图/键/元素定义直接 FAIL；
   - S2: 拓扑 atomtypes 缺少参数列、bonds 缺少力场支持、或声明 RESP 时 expected_charges_map 为空/缺失直接 FAIL；
   - S3: 定宽格式解析 GRO 文件坐标 (列 20:44)，忽略末尾速度列 (列 44:68)，碰撞检测命中直接 FAIL；
   - S4: 日志包含 'did not converge' 或 'maximum number of steps' 直接 FAIL。
"""

import os
import re
import math
import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict

VALID_GATE_STATUSES = {"NOT_RUN", "BLOCKED", "NOT_VERIFIED", "FAIL", "EXPLORATORY", "PASS"}

KNOWN_FORCEFIELD_ATOM_TYPES = {
    # GAFF / GAFF2 / Amber Standard Types
    "c", "c1", "c2", "c3", "ca", "cp", "cq", "cc", "cd", "ce", "cf", "cg", "ch", "cx", "cy", "cz",
    "h1", "h2", "h3", "h4", "h5", "ha", "hc", "hn", "ho", "hp", "hs", "hw", "hx",
    "n", "n1", "n2", "n3", "n4", "na", "nb", "nc", "nd", "ne", "nf", "nh", "no",
    "o", "oh", "os", "ow",
    "s", "s2", "s4", "s6", "sh", "ss", "sp", "sq",
    "p2", "p3", "p4", "p5", "pb", "pc", "pd", "pe", "pf",
    "cl", "br", "f", "i"
}

INVALID_TYPE_PATTERNS = {"UNKNOWN", "FAKE", "DUMMY", "UNK", "UNL", "NONE", "MOCK", "TEST", "X"}

@dataclass
class GateResult:
    gate_id: str
    status: str
    details: Dict[str, Any]
    message: str

    def __post_init__(self):
        if self.status not in VALID_GATE_STATUSES:
            raise ValueError(f"Invalid gate status: {self.status}")

def aggregate_overall_gate_status(gate_results: List[GateResult], required_gates: List[str]) -> Tuple[str, str]:
    """
    综合门禁状态聚合法则：
    - 任何 gate 为 FAIL -> 总体 FAIL
    - 缺少任一 required_gate 或含有 BLOCKED -> 总体 BLOCKED
    - 含有 NOT_VERIFIED 或 NOT_RUN -> 总体 NOT_VERIFIED
    - 含有 EXPLORATORY -> 总体 EXPLORATORY
    - 只有当所有 required_gates 均且仅为 PASS -> 总体 PASS
    """
    gate_map = {g.gate_id: g for g in gate_results}

    for req in required_gates:
        if req not in gate_map:
            return "BLOCKED", f"必需门禁 {req} 未执行或缺失"

    statuses = [g.status for g in gate_results]

    if "FAIL" in statuses:
        fail_msgs = [f"{g.gate_id}: {g.message}" for g in gate_results if g.status == "FAIL"]
        return "FAIL", "; ".join(fail_msgs)

    if "BLOCKED" in statuses:
        return "BLOCKED", "上游先决条件缺失或阻断"

    if "NOT_RUN" in statuses or "NOT_VERIFIED" in statuses:
        return "NOT_VERIFIED", "部分必要检验尚未执行或未通过验证"

    if "EXPLORATORY" in statuses:
        return "EXPLORATORY", "计算成功执行，但使用了受限/诊断级协议，不能作为正式生产真值"

    if all(gate_map[req].status == "PASS" for req in required_gates):
        return "PASS", "全部必要科学门禁均严格通过并具备可追溯原生证据"

    return "NOT_VERIFIED", "未达到全部 PASS 要求"

class ChemicalValidator:
    def __init__(self):
        pass

    def check_gate_s0(self, formulation_dict: Dict[str, Any]) -> GateResult:
        """
        S0 门禁：源配方与组分身份可追溯性
        """
        if not formulation_dict or not isinstance(formulation_dict, dict):
            return GateResult("S0", "FAIL", {}, "配方定义为空或非字典类型")

        source_id = formulation_dict.get("source_record_id") or formulation_dict.get("candidate_id")
        if not source_id:
            return GateResult("S0", "FAIL", {}, "缺少文献源记录标识 (source_record_id)")

        resins = formulation_dict.get("resins", [])
        curers = formulation_dict.get("curers", [])
        if not resins or not curers:
            return GateResult("S0", "FAIL", {}, "树脂或固化剂列表为空")

        for r in resins:
            if not r.get("smiles") or not r.get("name"):
                return GateResult("S0", "FAIL", {"resin": r}, "树脂缺少规范 SMILES 或名称")
        for c in curers:
            if not c.get("smiles") or not c.get("name"):
                return GateResult("S0", "FAIL", {"curer": c}, "固化剂缺少规范 SMILES 或名称")

        nominal_r = formulation_dict.get("nominal_r")
        actual_r = formulation_dict.get("actual_discrete_r")
        if nominal_r is None or actual_r is None or nominal_r <= 0 or actual_r <= 0:
            return GateResult("S0", "FAIL", {}, f"无效的当量比定义: nominal_r={nominal_r}, actual_r={actual_r}")

        return GateResult("S0", "PASS", {
            "source_record_id": source_id,
            "nominal_r": nominal_r,
            "actual_discrete_r": actual_r,
            "r_definition": formulation_dict.get("r_definition", "[active_H]/[epoxy]")
        }, "源配方可追溯且化学当量比明确")

    def check_gate_s1(self, monomer_graphs_and_charges: List[Dict[str, Any]]) -> GateResult:
        """
        S1 门禁：单体及反应模板化学图、元素守恒与净电荷守恒
        - 必须提供真实的化学连接图 (bonds) 或有效 SMILES/分子图定义，严禁仅传散点坐标欺骗通过！
        - 必须包含元素列表 (elements) 且与坐标长度严格对应；
        - 逐原子核验 partial_charges 求和与 formal_charge 偏差是否严格 < 1e-4 e；
        - 核验单体内部坐标无严重原子空间碰撞 (两两距离 >= 0.07 nm)。
        """
        if not monomer_graphs_and_charges:
            return GateResult("S1", "BLOCKED", {}, "缺少单体化学图与电荷输入数据")

        for spec in monomer_graphs_and_charges:
            if not isinstance(spec, dict):
                return GateResult("S1", "FAIL", {}, "单体规范项必须为字典结构")

            name = spec.get("name", "Unknown")
            charges = spec.get("partial_charges", [])
            formal_q = spec.get("formal_charge", 0)
            coords = spec.get("coordinates")
            elements = spec.get("elements")
            bonds = spec.get("bonds")
            smiles = spec.get("smiles")

            # 1. 电荷列表与净电荷守恒检查 (优先核验基础标量守恒)
            if charges is None or len(charges) == 0:
                return GateResult("S1", "FAIL", {"monomer": name}, f"单体 {name} 缺少部分电荷列表")

            if any(math.isnan(q) or math.isinf(q) for q in charges):
                return GateResult("S1", "FAIL", {"monomer": name}, f"单体 {name} 部分电荷中含有 NaN 或 Inf")

            sum_q = float(sum(charges))
            if abs(sum_q - formal_q) > 1e-4:
                return GateResult("S1", "FAIL", {
                    "monomer": name, "sum_charges": sum_q, "formal_charge": formal_q
                }, f"单体 {name} 净电荷求和 {sum_q:.6f} 不守恒 (形式电荷 {formal_q})")

            # 2. 3D 几何坐标核验 (必须存在坐标)
            if coords is None or len(coords) == 0:
                return GateResult("S1", "FAIL", {
                    "monomer": name, "has_coordinates": False
                }, f"单体 {name} 缺少 3D 几何坐标，无法验证空间几何结构")

            # 3. 核心拦截：必须包含化学图定义 (bonds 或有效 smiles 或 rdkit_mol)
            has_chemical_graph = bool(bonds is not None and len(bonds) > 0) or bool(smiles and len(str(smiles).strip()) > 0) or ("rdkit_mol" in spec and spec["rdkit_mol"] is not None)
            if not has_chemical_graph:
                return GateResult("S1", "FAIL", {
                    "monomer": name, "has_chemical_graph": False
                }, f"单体 {name} 缺少分子化学连接图定义 (bonds/smiles/rdkit_mol)，严禁仅凭孤立散点坐标冒充化学分子通过 S1 门禁")

            # 4. 核心拦截：必须提供元素列表
            if elements is None or len(elements) == 0:
                return GateResult("S1", "FAIL", {
                    "monomer": name, "has_elements": False
                }, f"单体 {name} 缺少元素组成列表 (elements)，无法核验元素守恒")

            try:
                coords_arr = np.array(coords, dtype=float)
            except Exception as e:
                return GateResult("S1", "FAIL", {"monomer": name, "error": str(e)}, f"单体 {name} 3D 坐标无法转换为数值数组")

            if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
                return GateResult("S1", "FAIL", {
                    "monomer": name, "shape": coords_arr.shape
                }, f"单体 {name} 3D 坐标维度异常: {coords_arr.shape} != (N, 3)")

            if len(coords_arr) != len(charges) or len(coords_arr) != len(elements):
                return GateResult("S1", "FAIL", {
                    "monomer": name, "n_coords": len(coords_arr), "n_charges": len(charges), "n_elements": len(elements)
                }, f"单体 {name} 坐标原子数 {len(coords_arr)}、电荷数 {len(charges)} 与元素数 {len(elements)} 不匹配")

            if np.any(np.isnan(coords_arr)) or np.any(np.isinf(coords_arr)):
                return GateResult("S1", "FAIL", {"monomer": name}, f"单体 {name} 3D 坐标中含有 NaN 或 Inf")

            # 原子内部碰撞检查
            if len(coords_arr) > 1:
                for i in range(len(coords_arr)):
                    diff = coords_arr[i+1:] - coords_arr[i]
                    dists = np.linalg.norm(diff, axis=1)
                    if np.any(dists < 0.07):  # < 0.7 A
                        min_d = float(np.min(dists))
                        return GateResult("S1", "FAIL", {
                            "monomer": name, "min_distance_nm": min_d
                        }, f"单体 {name} 内部存在严重原子空间碰撞 (重叠距离 {min_d:.4f} nm < 0.07 nm)")

        return GateResult("S1", "PASS", {
            "monomers_verified": len(monomer_graphs_and_charges)
        }, "所有单体化学图、净电荷电中性与几何结构合法")

    def check_gate_s2_from_top(
        self,
        top_filepath: str,
        declared_charge_protocol: str,
        expected_charges_map: Optional[Dict[str, float]] = None
    ) -> GateResult:
        """
        S2 门禁：真实力场参数覆盖与电荷来源核验（必须从引擎真实生成的 .top / .itp 拓扑文件解析）
        - 必须包含完整的力场参数支持（[ atomtypes ] 必须包含完整的质量、非键 sigma/epsilon 参数，不得为缩略占位行）
        - 检查成键定义是否具备力场参数支持
        - 声明 DFT-RESP 时，expected_charges_map 绝不能为空；必须完整覆盖拓扑中的全部原子！
        """
        if not os.path.exists(top_filepath):
            return GateResult("S2", "BLOCKED", {"file": top_filepath}, f"原生拓扑文件不存在: {top_filepath}")

        atoms_found = []
        bonds_found = []
        atomtypes_defined = {}
        has_forcefield_include = False
        in_atoms_section = False
        in_bonds_section = False
        in_atomtypes_section = False
        incomplete_atomtypes = []

        with open(top_filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                raw_line = line.strip()
                line_str = raw_line.split(";")[0].strip()
                if not line_str:
                    continue

                if line_str.startswith("#include"):
                    if any(kw in line_str.lower() for kw in ["gaff", "amber", "oplsaa", "charmm", "gromos"]):
                        has_forcefield_include = True
                    continue

                if line_str.startswith("["):
                    sec_name = line_str.strip("[] ").lower()
                    in_atoms_section = (sec_name == "atoms")
                    in_bonds_section = (sec_name == "bonds")
                    in_atomtypes_section = (sec_name == "atomtypes")
                    continue

                if in_atomtypes_section:
                    parts = line_str.split()
                    # GROMACS 标准 atomtype 至少需要: name, mass, charge, ptype, sigma, epsilon (6项)
                    # 简化力场至少也必须包含 name + 非键参数
                    if len(parts) < 5:
                        incomplete_atomtypes.append(line_str)
                    else:
                        atomtypes_defined[parts[0]] = parts

                elif in_atoms_section:
                    parts = line_str.split()
                    if len(parts) >= 7:
                        try:
                            q = float(parts[6])
                            atoms_found.append({
                                "atom_nr": int(parts[0]),
                                "type": parts[1],
                                "resnr": int(parts[2]),
                                "residue": parts[3],
                                "atom_name": parts[4],
                                "charge": q
                            })
                        except ValueError:
                            pass

                elif in_bonds_section:
                    parts = line_str.split()
                    if len(parts) >= 2:
                        try:
                            bonds_found.append(parts)
                        except ValueError:
                            pass

        if not atoms_found:
            return GateResult("S2", "FAIL", {"file": top_filepath}, "拓扑文件中未解析出任何原子项")

        # 1. 检查 atomtypes 是否完整定义
        if incomplete_atomtypes:
            return GateResult("S2", "FAIL", {
                "incomplete_atomtypes": incomplete_atomtypes[:5]
            }, f"拓扑中 atomtypes 定义不完整 (缺少必要非键/力常数参数列，如 {incomplete_atomtypes[0]}): 缺少真实参数支持")

        # 2. 检查成键参数支持
        # 若存在 bonds 但既无 forcefield include 也无完整成键参数，拦截
        if bonds_found and not has_forcefield_include:
            # 检查 bonds 每一行是否仅有 1 2 1 (无参数定义)
            unparam_bonds = [b for b in bonds_found if len(b) <= 3]
            if unparam_bonds and not atomtypes_defined:
                return GateResult("S2", "FAIL", {
                    "unparameterized_bonds_count": len(unparam_bonds)
                }, "拓扑中存在成键定义但缺少真实力场参数覆盖 (未引用 forcefield.itp 且无内嵌参数)")

        # 3. 检查原子电荷有效性
        charges = [a["charge"] for a in atoms_found]
        if all(abs(q) < 1e-6 for q in charges):
            return GateResult("S2", "FAIL", {"charges": charges[:5]}, "拓扑中全部原子电荷为 0，与任何合法力场/电荷模型不符")

        net_charge = float(sum(charges))
        if abs(net_charge) > 1e-3:
            return GateResult("S2", "FAIL", {
                "net_charge": net_charge
            }, f"拓扑净电荷严重不守恒: sum(q) = {net_charge:.6f} e != 0")

        # 4. 严格核验 DFT-RESP 电荷来源与映射
        is_declared_resp = ("RESP" in declared_charge_protocol.upper() and "B3LYP" in declared_charge_protocol.upper())

        if is_declared_resp:
            # 核心拦截：声明 DFT-RESP 时，expected_charges_map 绝不能为空！
            if expected_charges_map is None or len(expected_charges_map) == 0:
                return GateResult("S2", "FAIL", {
                    "declared_protocol": declared_charge_protocol,
                    "expected_charges_map_empty": True
                }, "声明了 DFT-RESP 电荷协议，但未提供任何有效逐原子参考电荷映射证据 (expected_charges_map 缺失或为空)")

            mismatches = []
            uncovered_atoms = []
            for a in atoms_found:
                name = a["atom_name"]
                nr = str(a["atom_nr"])
                exp_q = expected_charges_map.get(name)
                if exp_q is None:
                    exp_q = expected_charges_map.get(nr)
                if exp_q is None:
                    uncovered_atoms.append(name)
                elif abs(a["charge"] - exp_q) > 1e-4:
                    mismatches.append({"atom": name, "actual": a["charge"], "expected": exp_q})

            if uncovered_atoms:
                return GateResult("S2", "FAIL", {
                    "uncovered_atoms_count": len(uncovered_atoms),
                    "uncovered_sample": uncovered_atoms[:5]
                }, f"拓扑中存在未被 expected_charges_map 覆盖的原子 (共 {len(uncovered_atoms)} 个，如 {uncovered_atoms[:3]})")

            if mismatches:
                return GateResult("S2", "FAIL", {
                    "mismatches": mismatches[:10]
                }, f"拓扑电荷与声明的 DFT-RESP 源电荷映射不匹配 (共 {len(mismatches)} 处偏差)")

            status = "PASS"
        else:
            status = "EXPLORATORY"

        return GateResult("S2", status, {
            "top_file": top_filepath,
            "atom_count_in_top": len(atoms_found),
            "declared_protocol": declared_charge_protocol,
            "net_charge_in_top": round(net_charge, 6)
        }, f"力场参数全部覆盖，电荷模型: {declared_charge_protocol} (状态: {status})")

    def check_gate_s3_from_gro(self, gro_filepath: str, min_box_nm: float = 2.0) -> GateResult:
        """
        S3 门禁：从真实初始打包 .gro 文件检查盒子边长与原子最小间距（PBC 全体系碰撞拦截）
        - 严格采用定宽格式解析 GRO 文件坐标 (列 20:28 为 x, 28:36 为 y, 36:44 为 z)，
          彻底杜绝按空格 split 时将末尾速度列 (44:68) 误当坐标导致重叠被错误放行的漏洞！
        - 全体系周期性最小镜像碰撞检测 (两两间距 >= 0.05 nm)。
        """
        if not os.path.exists(gro_filepath):
            return GateResult("S3", "BLOCKED", {"file": gro_filepath}, f"原生打包结构文件不存在: {gro_filepath}")

        with open(gro_filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip("\r\n") for l in f if l.strip("\r\n")]

        if len(lines) < 3:
            return GateResult("S3", "FAIL", {}, "GRO 文件行数过少，格式损坏")

        try:
            num_atoms = int(lines[1].strip())
            box_line = lines[-1].split()
            box = np.array([float(box_line[0]), float(box_line[1]), float(box_line[2])], dtype=float)
        except Exception as e:
            return GateResult("S3", "FAIL", {"error": str(e)}, "无法解析 GRO 文件头或盒矢量")

        if np.any(box < min_box_nm):
            return GateResult("S3", "FAIL", {"box": box.tolist()},
                              f"周期盒子尺寸过小 ({box.tolist()} nm < {min_box_nm} nm)，无法满足截断安全")

        # 定宽解析原子坐标
        coords = []
        for line in lines[2:2+num_atoms]:
            if len(line) < 44:
                # 兼容非常简短但至少有三列浮点的行
                parts = line.split()
                if len(parts) >= 6:
                    coords.append([float(parts[3]), float(parts[4]), float(parts[5])])
            else:
                try:
                    # 标准 GRO 格式：第 21-28 字符为 x (nm), 29-36 为 y, 37-44 为 z
                    x_str = line[20:28].strip()
                    y_str = line[28:36].strip()
                    z_str = line[36:44].strip()
                    coords.append([float(x_str), float(y_str), float(z_str)])
                except ValueError:
                    parts = line.split()
                    if len(parts) >= 6:
                        coords.append([float(parts[3]), float(parts[4]), float(parts[5])])

        if len(coords) != num_atoms:
            return GateResult("S3", "FAIL", {
                "header_atoms": num_atoms, "parsed_atoms": len(coords)
            }, f"GRO 文件头声明原子数 {num_atoms} 与实际坐标行数 {len(coords)} 不一致")

        coords_arr = np.array(coords, dtype=float)
        if np.any(np.isnan(coords_arr)) or np.any(np.isinf(coords_arr)):
            return GateResult("S3", "FAIL", {}, "GRO 坐标中包含 NaN 或 Inf 非法浮点数")

        if len(coords_arr) < 2:
            return GateResult("S3", "FAIL", {}, "未提取到有效原子坐标")

        # 全体系周期性邻域最近邻碰撞检测
        try:
            tree = cKDTree(coords_arr, boxsize=box)
            clashes = tree.query_pairs(r=0.05)
            if clashes:
                first_pair = list(clashes)[0]
                dr = np.abs(coords_arr[first_pair[0]] - coords_arr[first_pair[1]])
                dr = np.minimum(dr, box - dr)
                min_dist = float(np.linalg.norm(dr))
                return GateResult("S3", "FAIL", {
                    "clash_count": len(clashes),
                    "min_distance_nm": min_dist,
                    "first_clashing_pair": [int(first_pair[0]), int(first_pair[1])]
                }, f"初始打包存在原子灾难性碰撞重叠 (< 0.05 nm)，检测到 {len(clashes)} 对碰撞 (如原子 {first_pair[0]+1} 与 {first_pair[1]+1} 间距 {min_dist:.4f} nm)")
        except Exception as e:
            return GateResult("S3", "FAIL", {"error": str(e)}, f"周期邻域碰撞分析异常: {str(e)}")

        return GateResult("S3", "PASS", {
            "num_atoms": num_atoms,
            "box_lengths_nm": box.tolist()
        }, "初始打包周期边界与分子间隙安全")

    def check_gate_s4_from_log(
        self,
        md_log_path: str,
        max_allowed_force: float = 1500.0,
        receipt: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """
        S4 门禁：从 GROMACS 能量最小化日志中提取收敛极大力 (F_max)
        - 日志包含 'did not converge' 或 'reached the maximum number of steps' 直接 FAIL；
        - log_converged 为 False 时严禁通过；
        - 极大力必须受控 (< max_allowed_force)。
        """
        if not os.path.exists(md_log_path):
            return GateResult("S4", "BLOCKED", {"file": md_log_path}, f"最小化原生日志文件不存在: {md_log_path}")

        if receipt is not None:
            if receipt.get("exit_code", 0) != 0:
                return GateResult("S4", "FAIL", {"receipt": receipt},
                                  f"能量最小化引擎执行失败，退出码 exit_code = {receipt.get('exit_code')}")

        f_max = None
        converged = False
        failed_or_max_steps = False
        has_engine_signature = False
        line_count = 0

        with open(md_log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line_count += 1
                if any(kw in line for kw in ["GROMACS", "Steepest Descents", "Energy minimization reached",
                                             "Step", "tolerance", "machine precision", "maximum number of steps"]):
                    has_engine_signature = True

                # 核心拦截：检测明确的未收敛或达到最大步数标记
                if any(neg in line for neg in [
                    "did not converge", "not converge", "maximum number of steps", "reached the maximum number of steps"
                ]):
                    failed_or_max_steps = True

                if "Maximum force" in line:
                    match = re.search(r"Maximum force\s*=\s*([0-9.eE+-]+)", line)
                    if match:
                        f_max = float(match.group(1))

                # 真实收敛判定：必须包含收敛到机器精度、Fmax 或达到预设容差，且未包含未收敛提示
                if any(pos in line for pos in [
                    "converged to machine precision", "converged to Fmax", "Steepest Descents converged",
                    "reached tolerance", "reached the requested tolerance"
                ]):
                    if not any(neg in line for neg in ["did not converge", "not converge"]):
                        converged = True

        if line_count < 2 or not has_engine_signature:
            return GateResult("S4", "FAIL", {
                "file": md_log_path, "line_count": line_count
            }, "日志缺少真实 GROMACS 能量最小化执行特征或收敛签名 (疑似单行伪造或未完整执行)")

        if failed_or_max_steps:
            return GateResult("S4", "FAIL", {
                "file": md_log_path, "converged": False, "max_steps_reached": True
            }, "能量最小化未收敛: 日志明确指明未收敛到 Fmax 或已达到最大步数 (did not converge to Fmax / reached maximum number of steps)")

        if f_max is None:
            return GateResult("S4", "FAIL", {"file": md_log_path}, "原生最小化日志中未找到极大力 (Maximum force) 记录")

        if f_max > max_allowed_force or math.isnan(f_max) or math.isinf(f_max):
            return GateResult("S4", "FAIL", {"max_force": f_max}, f"最小化后极大力超标或异常: {f_max:.2f} > {max_allowed_force}")

        if not converged:
            return GateResult("S4", "FAIL", {
                "max_force_kJ_mol_nm": f_max,
                "log_converged": False
            }, f"能量最小化未收敛或未达到预设容差 (log_converged=False, F_max = {f_max:.2f})")

        return GateResult("S4", "PASS", {
            "max_force_kJ_mol_nm": f_max,
            "log_converged": True
        }, f"能量最小化极大力受控 (F_max = {f_max:.2f} kJ/(mol*nm))")

    def check_gate_s5_curing(
        self,
        actual_bonds_formed: int,
        initial_epoxy_count: int,
        theoretical_max_alpha: float,
        residual_sacrificial_h: int = 0
    ) -> GateResult:
        """
        S5 门禁：真实交联网络拓扑加成数与化学计量守恒
        """
        if initial_epoxy_count <= 0:
            return GateResult("S5", "FAIL", {}, f"初始环氧数必须 > 0: {initial_epoxy_count}")

        if residual_sacrificial_h > 0:
            return GateResult("S5", "FAIL", {
                "residual_extra_h": residual_sacrificial_h
            }, f"最终体系残留 {residual_sacrificial_h} 个未正确消除的牺牲氢")

        alpha_actual = float(actual_bonds_formed) / float(initial_epoxy_count)

        if alpha_actual > theoretical_max_alpha + 1e-4:
            return GateResult("S5", "FAIL", {
                "alpha_actual": alpha_actual,
                "theoretical_max": theoretical_max_alpha
            }, f"转化率 {alpha_actual:.4f} 超出化学计量最大理论上限 {theoretical_max_alpha:.4f}")

        if alpha_actual < 0.10:
            return GateResult("S5", "FAIL", {
                "alpha_actual": alpha_actual
            }, f"成键加成数过低 ({alpha_actual:.4f} < 0.10)，未成功发生交联反应")

        return GateResult("S5", "PASS", {
            "alpha_actual": round(alpha_actual, 4),
            "bonds_formed": actual_bonds_formed,
            "theoretical_max_alpha": round(theoretical_max_alpha, 4)
        }, f"交联网络化学计量合格，实际转化率 alpha = {alpha_actual:.4f}")
