# 碳纤维复合材料智能预测平台 v1.3.1 (优化版)

## ✅ 本次优化（v1.3.1）

- **修复 TorchANI 力场特征全 0**：改为“同原子数分组批推理（无 padding）+ 片段独立计算再聚合”，并增加 `ani_n_atoms/ani_n_fragments/ani_success` 诊断列。
- **新增：SMILES 组分自动分列**：在“数据清洗 → SMILES组分分列”可一键把 `curing_agent_smiles` / `resin_smiles` 等列拆成 `*_1/*_2/*_key`，便于统计与类别平衡。
- **新增：3D 构象描述符**
- **新增（可选）：预训练 SMILES Transformer Embedding**：支持 ChemBERTa 等模型（需要安装 `transformers`，首次运行需下载权重）。：在“分子特征工程”中加入 `🧊 3D构象描述符 (RDKit3D+Coulomb)`，补充几何/构象层面的前沿表征。
- **增强：双组分（树脂+固化剂）融合**：除指纹/反应特征外，也可选择“拼接SMILES (Resin.Hardener)”让 RDKit/Mordred/3D/ANI/Transformer 等方法共同表征固化剂贡献。
- **新增：TDA 拓扑特征（持续同调 / Betti0-2）**：可将 3D 构象点云转为拓扑统计特征，用于表征网络孔洞/环路等结构信息。
- **新增：主动学习（Active Learning）页面**：基于不确定性 / EI / UCB 推荐下一批实验/模拟样本，形成数据-模型-实验闭环。

## 🚀 项目简介

本平台是一个基于机器学习的碳纤维复合材料（CFRP）性能预测系统

## ✨ 核心功能

### 📊 数据处理
- **智能数据清洗**: 缺失值处理、异常值检测、数据类型诊断修复
- **VAE数据增强**: 基于变分自编码器的表格数据生成
- **KNN智能填充**: 基于K近邻的缺失值预测

### 🧬 分子特征提取（5种方法）
- **RDKit标准版**: 200+分子描述符，适合中小型数据集
- **RDKit并行版**: 多进程加速，适合大数据集
- **RDKit内存优化版**: 分批处理，适合内存受限环境
- **Mordred描述符**: 1600+分子特征，最全面
- **图神经网络特征**: 分子拓扑结构特征

### 🤖 模型训练（15+模型 + 完整手动调参）
- **传统模型**: 线性回归、Ridge、Lasso、ElasticNet、SVR、决策树
- **集成模型**: 随机森林、Extra Trees、梯度提升树、AdaBoost
- **高级集成**: XGBoost、LightGBM、CatBoost
- **深度学习**: 自定义神经网络(ANN)
- **AutoML**: TabPFN、AutoGluon
- **手动调参界面**: 可视化配置所有模型超参数

### 📈 模型解释
- **SHAP分析**: 特征重要性可视化
- **学习曲线**: 模型收敛分析
- **残差分析**: 预测误差分布
- **适用域分析**: PCA凸包边界检测

### ⚙️ 智能优化
- **Optuna超参数优化**: 贝叶斯优化调参
- **逆向设计**: 根据目标值反推输入参数

## 📁 项目结构

```
ML_CFRP-Resin_v1.2.10/
├── app.py                    # 主应用入口（完整11个页面）
├── config.py                 # 全局配置
├── requirements.txt          # 依赖列表
├── generate_sample_data.py   # 示例数据生成
├── README.md                 # 说明文档
│
├── core/                     # 核心模块
│   ├── __init__.py
│   ├── data_processor.py     # 数据清洗与增强
│   ├── data_explorer.py      # 数据探索与可视化
│   ├── model_trainer.py      # 模型训练器
│   ├── model_interpreter.py  # 模型解释（SHAP等）
│   ├── molecular_features.py # 分子特征提取（5种方法）
│   ├── graph_utils.py        # 图神经网络工具
│   ├── feature_selector.py   # 特征选择（完整UI）
│   ├── optimizer.py          # 超参数优化
│   ├── visualizer.py         # 可视化工具
│   ├── applicability_domain.py # 适用域分析
│   ├── ann_model.py          # 神经网络模型
│   └── ui_config.py          # UI配置与手动调参
│
└── datasets/                 # 数据目录
```

## 🛠️ 安装

### 1. 创建环境

```bash
conda create -n CFRP_env python=3.10
conda activate CFRP_env
```

### 2. 安装PyTorch

```bash
# CPU版本
pip install torch torchvision torchaudio

# GPU版本（CUDA 11.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 安装PyTorch Geometric（可选）

```bash
pip install torch_geometric
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
```

## 🚀 运行

```bash
streamlit run app.py
```

## 📋 完整功能页面

| 页面 | 功能 |
|------|------|
| 🏠 首页 | 功能介绍、快速开始 |
| 📤 数据上传 | 上传CSV/Excel、生成示例数据 |
| 🔍 数据探索 | 统计分析、相关性、分布图、缺失值 |
| 🧹 数据清洗 | 缺失值处理、异常值检测、重复数据 |
| ✨ 数据增强 | KNN填充、VAE生成 |
| 🧬 分子特征 | **5种提取方法选择** |
| 🎯 特征选择 | **完整UI：方差筛选、相关性筛选、智能推荐** |
| 🤖 模型训练 | **完整手动调参界面** |
| 📊 模型解释 | SHAP、学习曲线、特征重要性 |
| 🔮 预测应用 | 手动输入、批量预测、适用域分析 |
| ⚙️ 超参优化 | Optuna智能优化 |


## 📄 许可证

MIT License
