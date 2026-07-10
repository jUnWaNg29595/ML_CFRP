# 碳纤维复合材料智能预测平台 v1.4.5

## 🚀 项目简介

本平台是一个基于机器学习的碳纤维复合材料（CFRP）性能预测系统

## ⚙️ v1.4.5 更新：OpenMP 线程优化

解决了 RDKit 底层 OpenMP 占满所有 CPU 核心的问题。

**问题描述：**
即使设置 `n_jobs=1`，RDKit 内部的 OpenMP 仍会启动与 CPU 核心数相等的线程，导致 CPU 占用率 100%。

**解决方案：**
- 新增 `core/thread_config.py` 模块，在导入 RDKit 之前设置环境变量
- 默认限制线程数为 CPU 核心数的一半（最多 8 个）

**自定义线程数：**
```bash
# 方法1：设置环境变量
export ML_THREAD_COUNT=4
streamlit run app.py

# 方法2：直接设置 OpenMP 线程数
export OMP_NUM_THREADS=4
streamlit run app.py
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


## AMD Integrated GPU on WSL (Ubuntu 22.04)

This project uses PyTorch for GPU acceleration. On AMD GPUs, PyTorch uses the ROCm/HIP backend but **still exposes the `torch.cuda` API**, so in the UI you should choose device `cuda` to use GPU.

High level steps (refer to AMD official docs for exact versions):
1) Install the compatible AMD Windows driver for WSL.
2) In WSL Ubuntu 22.04, install ROCm using `amdgpu-install` with `--usecase=wsl,rocm --no-dkms`.
3) Install ROCm/HIP PyTorch wheels from AMD repo (`repo.radeon.com`).
4) Verify: `rocminfo` and `python -c "import torch; print(torch.cuda.is_available(), torch.version.hip, torch.cuda.get_device_name(0))"`.

