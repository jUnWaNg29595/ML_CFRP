#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整诊断脚本 - 检查所有可能的问题
"""

import os
import sys

# 设置环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("=" * 70)
print("ChemBERTa 完整诊断")
print("=" * 70)

# 1. 检查缓存目录
print("\n[1] 检查缓存目录")
cache_base = os.path.expanduser("~/.cache/huggingface/hub")
model_cache_name = "models--seyonec--ChemBERTa-zinc-base-v1"
model_cache_dir = os.path.join(cache_base, model_cache_name)

print(f"缓存目录: {model_cache_dir}")
print(f"存在: {os.path.exists(model_cache_dir)}")

if os.path.exists(model_cache_dir):
    # 检查 refs/main
    refs_main = os.path.join(model_cache_dir, "refs", "main")
    if os.path.exists(refs_main):
        with open(refs_main, 'r') as f:
            snapshot_id = f.read().strip()
        print(f"快照 ID: {snapshot_id}")

        snapshot_path = os.path.join(model_cache_dir, "snapshots", snapshot_id)
        print(f"快照路径: {snapshot_path}")
        print(f"快照存在: {os.path.exists(snapshot_path)}")

        if os.path.exists(snapshot_path):
            files = os.listdir(snapshot_path)
            print(f"快照文件: {', '.join(files)}")

            # 检查必需文件
            required_files = ['config.json', 'tokenizer_config.json', 'vocab.json']
            model_files = ['pytorch_model.bin', 'model.safetensors']

            missing = [f for f in required_files if f not in files]
            has_model = any(f in files for f in model_files)

            if missing:
                print(f"[WARNING] 缺少文件: {', '.join(missing)}")
            else:
                print("[OK] 所有必需文件都存在")

            if not has_model:
                print("[WARNING] 缺少模型权重文件")
            else:
                print("[OK] 模型权重文件存在")

    # 检查 .no_exist 目录
    no_exist_dir = os.path.join(model_cache_dir, ".no_exist")
    if os.path.exists(no_exist_dir):
        print(f"[WARNING] 发现 .no_exist 目录: {no_exist_dir}")
        print("这可能导致 transformers 尝试重新下载")
    else:
        print("[OK] 没有 .no_exist 目录")

# 2. 测试环境变量
print("\n[2] 检查环境变量")
env_vars = {
    'HF_ENDPOINT': os.environ.get('HF_ENDPOINT'),
    'HF_HUB_OFFLINE': os.environ.get('HF_HUB_OFFLINE'),
    'TRANSFORMERS_OFFLINE': os.environ.get('TRANSFORMERS_OFFLINE'),
    'USE_TF': os.environ.get('USE_TF'),
    'USE_TORCH': os.environ.get('USE_TORCH'),
}
for key, value in env_vars.items():
    print(f"{key}: {value}")

# 3. 测试加载
print("\n[3] 测试加载模型")
try:
    from core.molecular_features import SmilesTransformerEmbeddingExtractor

    print("初始化提取器...")
    extractor = SmilesTransformerEmbeddingExtractor(
        model_name="seyonec/ChemBERTa-zinc-base-v1",
        pooling="cls"
    )
    print("[OK] 初始化成功")

    print("\n提取特征...")
    df, names = extractor.smiles_to_embeddings(["CCO"], batch_size=1)
    print(f"[OK] 特征提取成功: {df.shape}")

except Exception as e:
    print(f"[ERROR] 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("诊断完成")
print("=" * 70)
