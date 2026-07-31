# CFRP 智能预测平台

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![Version](https://img.shields.io/badge/Version-1.5.1-blueviolet.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**基于机器学习的碳纤维复合材料性能预测与虚拟筛选系统**

[English](#english) | [中文文档](#中文文档)

</div>

---

## 中文文档

### 📋 目录

- [项目简介](#项目简介)
- [核心功能](#核心功能)
- [技术架构](#技术架构)
- [快速开始](#快速开始)
- [使用指南](#使用指南)
- [性能优化](#性能优化)
- [常见问题](#常见问题)

---

### 项目简介

本平台是一个集成的机器学习系统，用于碳纤维增强聚合物（CFRP）的性能预测与材料筛选。系统结合了分子特征工程、深度学习模型、虚拟筛选和主动学习等先进技术。

**当前版本：`1.5.1`（2026-07-31）**

本版本在多步骤分子特征工作流复现、筛选前特征映射确认、大规模候选筛选稳定性和工艺特征 PLS 工作流的基础上，修复了导入分子特征流程时 Torch 局部作用域导致的页面崩溃问题。

**主要应用场景:**
- 🎯 **性能预测**: 预测CFRP的拉伸强度、模量等关键性能指标
- 🔬 **虚拟筛选**: 高通量生成候选分子并筛选配方
- 🧪 **分子特征提取**: 从SMILES自动生成分子描述符
- 📊 **模型解释**: SHAP可视化、特征重要性分析
- 🤖 **主动学习**: 智能推荐最有价值的实验样本

---

### 核心功能

#### 1. 🧪 虚拟分子筛选 (核心亮点)
- **反应约束生成**: 基于环氧树脂化学反应规则生成候选配方
- **PubChem集成**: 自动检索PubChem数据库扩充候选库
- **多源候选融合**: 组合树脂库、固化剂库、PubChem候选
- **化学规则过滤**: 环氧官能度、分子量、芳香环数、元素组成
- **可合成性评估**: 合成难度评分与可行性筛选
- **配方可行性**: 当量比计算、组分配比合理性检查
- **适用域分析**: 预测结果的可信度评估
- **筛选前特征确认**: 仅对需要改变的模型特征进行人工映射，未选择的特征保留候选配方已有值
- **严格输入校验**: 对缺失、非数值和无穷特征进行明确检查，避免无效矩阵进入模型预测

#### 2. 🧬 分子特征工程
**指纹特征:**
- Morgan指纹 (ECFP)
- MACCS keys
- RDKit指纹
- Atom Pair指纹

**描述符特征:**
- RDKit描述符 (210+)
- Mordred描述符 (1800+)
- 3D分子描述符

**深度学习特征:**
- ChemBERTa (Transformer预训练模型)
- 分子图神经网络 (GNN)
- SMILES Transformer

**工作流复现:**
- 支持在已有训练流程上追加新的分子特征提取步骤
- 保存完整的步骤顺序、来源列、有效行映射和最终特征名
- 筛选阶段可依据训练工作流重建模型输入特征

#### 3. 🤖 模型训练
**支持的模型:**
| 模型类型 | 特点 | 适用场景 |
|---------|------|----------|
| **XGBoost** | 快速、高效、可解释 | 中小数据集 |
| **BNN (贝叶斯神经网络)** | 不确定性量化 | 需要置信度的场景 |
| **PINN (物理信息神经网络)** | 融入物理约束 | 物理规律明确的系统 |
| **TabNet** | 深度表格学习 | 大规模表格数据 |
| **GNN (图神经网络)** | 分子图结构学习 | 分子结构重要场景 |
| **Transformer系列** | 注意力机制 | 序列建模 |
| **AutoGluon** | 自动化集成 | 快速建模 |

**高级功能:**
- 超参数优化 (Optuna)
- 交叉验证
- 早停机制
- 模型集成
- GPU加速训练
- 工艺特征 PLS：在特征选择页锁定稀疏工艺特征降维 workflow，训练和筛选阶段复用同一套无泄漏 Pipeline

#### 4. 📊 模型解释与可视化
- **SHAP分析**: 特征重要性、依赖图、交互图
- **学习曲线**: 训练过程可视化
- **预测分析**: 残差图、误差分布
- **特征相关性**: 相关性矩阵、散点图

#### 5. 🔍 主动学习
- **不确定性采样**: 选择模型最不确定的样本
- **多样性采样**: 选择最具代表性的样本
- **混合策略**: 综合不确定性与多样性
- **批量推荐**: 批量推荐最有价值的实验

#### 6. 🖼️ 图像转SMILES
- **DECIMER集成**: 从化学结构图像识别SMILES
- **多格式支持**: PNG, JPG, PDF等
- **批量处理**: 支持批量图像识别

---

### 技术架构

```
CFRP系统/
├── app.py                      # Streamlit主应用
├── config.py                   # 全局配置
├── UserPrediction.py           # 用户预测模块
│
├── core/                       # 核心功能模块
│   ├── virtual_screening.py    # 虚拟筛选 (核心)
│   ├── molecular_features.py   # 分子特征提取
│   ├── process_pls.py          # 工艺特征PLS工作流
│   ├── model_trainer.py        # 模型训练
│   ├── model_interpreter.py    # 模型解释 (SHAP)
│   ├── pubchem_client.py       # PubChem API
│   ├── smiles_utils.py         # SMILES处理
│   ├── epoxy_physics.py        # 环氧物理约束
│   ├── active_learning.py      # 主动学习
│   └── ...                     # 其他模块
│
├── DECIMER/                    # 图像转SMILES
│   ├── decimer.py
│   └── efficientnetv2/
│
├── docs/                       # 文档
└── scripts/                    # 辅助脚本
```

---

### 快速开始

#### 1. 环境准备

**系统要求:**
- Python 3.10+
- 推荐: 8GB+ RAM, 4核+ CPU
- GPU训练: NVIDIA GPU with CUDA 11.8+

**创建虚拟环境:**
```bash
conda create -n cfrp_env python=3.10
conda activate cfrp_env
```

#### 2. 安装依赖

**基础依赖:**
```bash
# PyTorch (CPU版本)
pip install torch torchvision torchaudio

# 核心依赖
pip install streamlit pandas numpy scikit-learn
pip install rdkit mordred
pip install xgboost lightgbm
pip install shap optuna

# Web应用
pip install plotly altair
```

**GPU加速 (可选):**
```bash
# PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# GNN支持
pip install torch_geometric
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
```

**图像转SMILES (可选):**
```bash
pip install tensorflow>=2.12.0,<=2.20.0
pip install opencv-python pillow-heif efficientnet selfies pyyaml
pip install pymupdf  # PDF支持
```

#### 3. 启动应用

```bash
streamlit run app.py
```

浏览器自动打开: `http://localhost:8501`

---

### 使用指南

#### 完整工作流程

```
1. 数据上传 → 2. 数据清洗 → 3. 分子特征提取 → 4. 特征选择
      ↓              ↓                ↓                 ↓
   CSV/XLSX    缺失值处理      SMILES→描述符        降维/筛选
                                  ↓
                            5. 模型训练 ← 6. 超参数优化
                                  ↓
                            7. 模型解释
                                  ↓
                            8. 预测/虚拟筛选
                                  ↓
                            9. 主动学习
```

#### 虚拟筛选使用流程

1. **准备模型和特征配置**
   - 训练模型并导出 `.joblib` 文件
   - 保存分子特征流程配置 `feature_process.json`

2. **上传到虚拟筛选页面**
   - 导入训练好的模型
   - 上传分子特征配置文件

3. **配置候选库**
   - 上传树脂SMILES列表 (CSV)
   - 上传固化剂SMILES列表 (可选)
   - 设置PubChem搜索关键词 (可选)

4. **设置筛选参数**
   - 总候选数上限
   - 化学规则过滤参数
   - 目标性能范围
   - 不确定度阈值

5. **运行筛选**
   - 点击"开始虚拟筛选"
   - 查看实时进度
   - 导出筛选结果

#### 关键参数说明

**化学规则过滤:**
- `min_epoxide`: 最小环氧官能度 (建议: 1-2)
- `min_aromatic_rings`: 最小芳香环数 (建议: 1-2)
- `min_mw`: 最小分子量 (建议: 180-250)
- `allowed_elements`: 允许的元素组成

**配方可行性:**
- `amine_ratio`: 胺当量比范围 (建议: 0.25-4.0)
- `reject_mixed_class`: 是否拒绝混合类型固化剂

**性能优化:**
- `n_jobs`: 并行任务数 (建议: CPU核心数的50%)
- `batch_size`: 批量预测大小
- `use_gpu`: 是否使用GPU加速

---

### 性能优化

#### OpenMP线程优化 (v1.4.5)

**问题:** RDKit底层OpenMP会占满所有CPU核心，即使设置`n_jobs=1`

**解决方案:**
系统自动限制线程数为CPU核心数的一半(最多8个)

**自定义线程数:**
```bash
# 方法1: 环境变量
export ML_THREAD_COUNT=4
streamlit run app.py

# 方法2: OpenMP设置
export OMP_NUM_THREADS=4
streamlit run app.py
```

#### GPU内存优化

```python
# 在app.py中启用GPU内存增长
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

#### 大规模数据集

- 使用`batch_size`参数分批预测
- 启用特征缓存
- 考虑使用更快的模型(XGBoost > 神经网络)

---

### 常见问题

**Q1: 虚拟筛选后候选为空?**
- 检查化学规则过滤参数是否过于严格
- 降低`min_epoxide`和`min_aromatic_rings`
- 尝试关闭化学规则过滤(`filter_timing='off'`)

**Q2: 模型导入失败?**
- 确认模型文件来自本系统导出
- 检查依赖版本是否一致
- 查看错误日志定位具体模块

**Q3: GPU训练报错?**
- 确认CUDA版本与PyTorch匹配
- 检查GPU内存是否足够
- 尝试减小batch_size

**Q4: 特征提取过慢?**
- 减少特征类型(如不使用Mordred)
- 启用特征缓存
- 使用`n_jobs`并行化

**Q5: SHAP计算耗时过长?**
- 减少背景样本数(`background_samples`)
- 使用TreeExplainer(仅限树模型)
- 降低可视化样本数

---

## English

### Project Overview

An integrated machine learning system for Carbon Fiber Reinforced Polymer (CFRP) performance prediction and virtual screening. The system combines molecular feature engineering, deep learning models, virtual screening, and active learning.

### Key Features

- **Virtual Screening**: Reaction-constrained candidate generation with PubChem integration
- **Molecular Features**: Fingerprints, RDKit/Mordred descriptors, ChemBERTa embeddings
- **Multiple Models**: XGBoost, BNN, PINN, TabNet, GNN, Transformer, AutoGluon
- **Model Interpretation**: SHAP analysis, feature importance
- **Active Learning**: Smart sample recommendation

### Quick Start

```bash
# Create environment
conda create -n cfrp_env python=3.10
conda activate cfrp_env

# Install dependencies
pip install streamlit pandas numpy scikit-learn rdkit xgboost shap
pip install torch torchvision

# Run application
streamlit run app.py
```

### Project Structure

```
CFRP系统/
├── app.py                      # Streamlit main app
├── core/                       # Core modules
│   ├── virtual_screening.py    # Virtual screening
│   ├── molecular_features.py   # Feature extraction
│   ├── model_trainer.py        # Model training
│   └── ...
└── docs/                       # Documentation
```

### License

MIT License

---

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

<div align="center">

**⭐ If this project helps your research, please give it a star! ⭐**

</div>
