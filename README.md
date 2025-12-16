# 碳纤维复合材料智能预测平台 v1.3.4

## 🚀 项目简介

本平台是一个基于机器学习的碳纤维复合材料（CFRP）性能预测系统

## 📁 项目结构

```
ML_CFRP-Resin_v1.3.4/
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
## 📄 许可证

MIT License
