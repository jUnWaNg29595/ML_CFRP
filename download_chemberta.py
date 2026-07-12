#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手动下载 ChemBERTa 模型到本地
使用方法：python download_chemberta.py
"""

import os
import sys

# 禁用 TensorFlow（必须在导入 transformers 之前设置）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

# 设置控制台编码为 UTF-8（Windows 兼容）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 设置镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoTokenizer, AutoConfig

def download_model():
    """下载 ChemBERTa 模型到本地缓存"""
    model_name = "seyonec/ChemBERTa-zinc-base-v1"
    local_dir = "./models/ChemBERTa-zinc-base-v1"

    print(f"正在下载模型: {model_name}")
    print(f"保存位置: {local_dir}")
    print("=" * 60)

    try:
        # 下载 tokenizer
        print("1. 下载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=local_dir
        )
        print("[OK] Tokenizer 下载完成")

        # 下载配置文件
        print("\n2. 下载配置文件...")
        config = AutoConfig.from_pretrained(
            model_name,
            cache_dir=local_dir
        )
        print(f"[OK] 配置文件下载完成 (模型类型: {config.model_type})")

        # 根据配置动态加载正确的模型类
        print("\n3. 下载模型权重...")
        # 使用 AutoModel 而不是 RobertaModel，避免 TensorFlow 导入
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=local_dir
        )
        print("[OK] 模型下载完成")

        print("\n" + "=" * 60)
        print("[SUCCESS] 所有文件下载成功！")
        print(f"\n使用方法：在代码中将 model_name 改为: '{local_dir}'")
        print(f"或者直接使用原始名称，模型已缓存到本地")

    except Exception as e:
        print(f"\n[ERROR] 下载失败: {e}")
        print("\n备选方案：")
        print("1. 检查网络连接")
        print("2. 尝试使用 VPN")
        print("3. 从 https://hf-mirror.com/seyonec/ChemBERTa-zinc-base-v1 手动下载")
        print("4. 升级 transformers: pip install --upgrade transformers")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    download_model()
