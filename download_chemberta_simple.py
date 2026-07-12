#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版 ChemBERTa 模型下载脚本
避免 TensorFlow 导入冲突
"""

import os
import sys

# 禁用 TensorFlow（避免与 transformers 冲突）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

# 设置镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 设置控制台编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def download_model():
    """下载 ChemBERTa 模型"""
    model_name = "seyonec/ChemBERTa-zinc-base-v1"

    print(f"正在下载模型: {model_name}")
    print("=" * 60)

    try:
        # 只导入 PyTorch 版本
        from transformers import AutoTokenizer, AutoModel
        import torch

        print("1. 下载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("[OK] Tokenizer 下载完成")

        print("\n2. 下载模型权重...")
        model = AutoModel.from_pretrained(model_name)
        print("[OK] 模型下载完成")

        print("\n" + "=" * 60)
        print("[SUCCESS] 模型已缓存到本地！")
        print("\n现在可以正常运行 app.py 了")

    except Exception as e:
        print(f"\n[ERROR] 下载失败: {e}")
        print("\n如果仍然超时，请尝试：")
        print("1. 使用 VPN 或代理")
        print("2. 手动从 https://hf-mirror.com/seyonec/ChemBERTa-zinc-base-v1 下载")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    download_model()
