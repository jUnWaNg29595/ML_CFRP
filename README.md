# 碳纤维复合材料智能预测平台 v1.4.2

## 🚀 项目简介

本平台是一个基于机器学习的碳纤维复合材料（CFRP）性能预测系统

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


## 🖼️ 图像/文件转 SMILES（DECIMER）

平台已集成 **DECIMER**（Image Transformer）用于从化学结构图像识别 SMILES。

- 入口：侧边栏 **“🖼️ 图像转SMILES”**
- 支持：png/jpg/jpeg/bmp/tif/tiff/webp/heif/heic；PDF（需安装 PyMuPDF 或 pdf2image）
- 注意：**首次运行会自动下载预训练权重（需要联网）**

### 安装依赖（可选）

```bash
pip install tensorflow>=2.12.0,<=2.20.0
pip install opencv-python pystow pillow-heif efficientnet selfies pyyaml
# 若需要 PDF 支持（二选一）
pip install pymupdf
# 或：pip install pdf2image  （系统需额外安装 poppler）
```
