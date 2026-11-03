"""
真实网络拓扑特征提取器 (Rigorous Topology Extractor)
科学准则：
1. 严禁将全原子图中度数 >= 3 的重原子误判为交联节点（苯环碳、主链支化碳度数均为3，会导致灾难性统计偏差）；
2. 严禁使用“质量乘 14”或“默认 350”等任何经验硬编码伪造链段分子量；
3. 严格基于真实形成的交联共价键（C-N 开环加成边）；
4. 真实反应转化率 md_alpha_actual = 实际生成加成键数 / 初始环氧总数；
5. 建立单体收缩图（Monomer Contracted Graph），同一对单体多次反应保留多重边；
6. 严格弹性有效链段判据若未完全实现，输出 None (null) 并标为 NOT_VERIFIED，绝不伪造。
"""

import math
from typing import Dict, Any, List, Set, Tuple, Optional
import networkx as nx

class TopologyExtractor:
    def __init__(self):
        pass

    def extract_from_covalent_network(
        self,
        monomer_units: List[Dict[str, Any]],
        reaction_crosslink_bonds: List[Tuple[int, int]],
        initial_epoxy_count: int,
        box_volume_nm3: float
    ) -> Dict[str, Any]:
        """
        基于真实单体列表与反应生成的共价加成键提取拓扑特征
        :param monomer_units: 体系中每个单体的字典列表，含 index, name, type ("epoxy"/"hardener"), mw, atom_indices
        :param reaction_crosslink_bonds: 实际形成的交联键列表 [(monomer_u_idx, monomer_v_idx), ...]
        :param initial_epoxy_count: 初始环氧基团总数
        :param box_volume_nm3: 平衡盒子真实体积 (nm3)
        """
        if initial_epoxy_count <= 0:
            raise ValueError(f"初始环氧数必须 > 0: {initial_epoxy_count}")

        # 1. 实际反应转化率 (基于真实加成事件数)
        num_bonds_formed = len(reaction_crosslink_bonds)
        alpha_actual = float(num_bonds_formed) / float(initial_epoxy_count)

        # 2. 建立单体级收缩图 (Monomer Contraction Graph, MultiGraph)
        G_mono = nx.MultiGraph()
        total_mass = 0.0
        for m in monomer_units:
            idx = m["index"]
            mw = float(m.get("mw", 0.0))
            total_mass += mw
            G_mono.add_node(idx, name=m.get("name", ""), type=m.get("type", ""), mw=mw)

        for u, v in reaction_crosslink_bonds:
            G_mono.add_edge(u, v)

        # 3. 最大连通团簇（Percolated Gel 质量占比）
        components = list(nx.connected_components(G_mono))
        cluster_masses = [
            sum(G_mono.nodes[n]["mw"] for n in comp)
            for comp in components
        ]
        largest_cluster_mass = max(cluster_masses) if cluster_masses else 0.0
        largest_cluster_fraction = (largest_cluster_mass / total_mass) if total_mass > 0 else 0.0

        # 4. 真实化学交联节点（连接了 3 个及以上反应臂的单体，或度数 >= 3 的单体）
        junction_monomers = [n for n in G_mono.nodes() if G_mono.degree(n) >= 3]
        num_junctions = len(junction_monomers)

        N_A = 6.02214076e23
        V_cm3 = box_volume_nm3 * 1e-21
        junction_density_mol_cm3 = (num_junctions / N_A) / V_cm3 if V_cm3 > 0 else 0.0

        # 5. 悬垂支链比例 (Dangling Chains)
        # 单体图中仅与 1 个其他单体连接的端基支链
        dangling_count = sum(1 for n in G_mono.nodes() if G_mono.degree(n) == 1)
        total_reactive_nodes = sum(1 for n in G_mono.nodes() if G_mono.degree(n) > 0)
        dangling_chain_fraction = (float(dangling_count) / float(total_reactive_nodes)) if total_reactive_nodes > 0 else 0.0

        # 6. 单体拓扑环秩 (Cycle Rank = E - V + C)
        e = G_mono.number_of_edges()
        v = G_mono.number_of_nodes()
        c = len(components)
        cycle_rank = max(0, e - v + c)

        # 7. 弹性有效链段分子量 (Elastic Strand MW)
        # 严格高分子物理声明：由于当前尚未实现结合仿射/幻象网络模型的分支链段精确张力解耦，
        # 绝不使用经验硬编码（如质量乘14或固定350）；严格置为 null 并标为 NOT_VERIFIED
        strand_mw_mean = None
        strand_verification = "NOT_VERIFIED"

        return {
            "md_alpha_actual": round(float(alpha_actual), 4),
            "md_largest_cluster_fraction": round(float(largest_cluster_fraction), 4),
            "md_junction_density_mol_cm3": round(float(junction_density_mol_cm3), 6),
            "md_strand_mw_mean_g_mol": strand_mw_mean,
            "md_strand_verification_status": strand_verification,
            "md_dangling_chain_fraction": round(float(dangling_chain_fraction), 4),
            "md_cycle_rank": int(cycle_rank),
            "qc_actual_crosslink_bonds": int(num_bonds_formed),
            "qc_junction_monomer_count": int(num_junctions),
            "qc_connected_components": int(len(components))
        }
