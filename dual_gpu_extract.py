"""双 GPU 并行提取 GNN 特征"""
import pandas as pd
import numpy as np
from core.graph_utils import GNNFeaturizer
import torch
from concurrent.futures import ThreadPoolExecutor
import time

def extract_gnn_features_dual_gpu(smiles_list, config):
    """使用双 GPU 并行提取特征"""

    # [新增] 启动前检查并清理显存
    if torch.cuda.is_available():
        print("\n检查 GPU 显存状态...")
        for i in range(min(2, torch.cuda.device_count())):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            total = props.total_memory / 1024**3
            free = total - allocated

            print(f"GPU {i}: {allocated:.2f} GB / {total:.2f} GB (可用: {free:.2f} GB)")

            if free < 8:
                print(f"⚠️ GPU {i} 可用显存不足 8GB，正在清理...")
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                allocated = torch.cuda.memory_allocated(i) / 1024**3
                free = total - allocated
                print(f"   清理后: {allocated:.2f} GB / {total:.2f} GB (可用: {free:.2f} GB)")

                if free < 6:
                    print(f"❌ GPU {i} 显存仍然不足，建议:")
                    print("   1. 运行 emergency_gpu_cleanup.py")
                    print("   2. 关闭其他占用 GPU 的程序")
                    print("   3. 降低 Batch Size")
                    return None, None

    # 分割数据
    mid = len(smiles_list) // 2
    smiles_gpu0 = smiles_list[:mid]
    smiles_gpu1 = smiles_list[mid:]

    print(f"\n总样本数: {len(smiles_list)}")
    print(f"GPU 0 处理: {len(smiles_gpu0)} 样本")
    print(f"GPU 1 处理: {len(smiles_gpu1)} 样本")

    results = [None, None]

    def extract_on_gpu(gpu_id, smiles_subset, idx):
        """在指定 GPU 上提取特征"""
        try:
            # [修复] 在初始化前先清理该 GPU 的显存
            if torch.cuda.is_available():
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

            # 提取 featurize 方法的参数
            batch_size = config.get('batch_size', 32)
            chunk_size = config.get('chunk_size', 512)
            num_workers = config.get('num_workers', 4)

            # 配置指定 GPU
            gpu_config = config.copy()
            # 移除 featurize 的参数
            gpu_config.pop('batch_size', None)
            gpu_config.pop('chunk_size', None)
            gpu_config.pop('num_workers', None)
            gpu_config['device'] = torch.device(f'cuda:{gpu_id}')

            # [关键修复] 禁用图缓存，防止内存泄漏
            gpu_config['cache_graphs'] = False
            gpu_config['max_cache_size'] = 0

            print(f"GPU {gpu_id} 开始提取...")
            start = time.time()

            featurizer = GNNFeaturizer(**gpu_config)
            features, valid_idx = featurizer.featurize(
                smiles_subset,
                batch_size=batch_size,
                chunk_size=chunk_size,
                num_workers=num_workers
            )

            elapsed = time.time() - start
            print(f"GPU {gpu_id} 完成! 耗时: {elapsed:.1f}s, 速度: {len(smiles_subset)/elapsed:.1f} 分子/秒")

            results[idx] = (features, valid_idx)

            # [修复] 提取完成后立即清理显存和内存
            del featurizer
            import gc
            gc.collect()
            if torch.cuda.is_available():
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"GPU {gpu_id} 错误: {e}")
            import traceback
            traceback.print_exc()
            results[idx] = None

    # 并行执行
    with ThreadPoolExecutor(max_workers=2) as executor:
        future0 = executor.submit(extract_on_gpu, 0, smiles_gpu0, 0)
        future1 = executor.submit(extract_on_gpu, 1, smiles_gpu1, 1)

        future0.result()
        future1.result()

    # 合并结果
    if results[0] is None or results[1] is None:
        print("❌ 部分 GPU 提取失败")
        return None, None

    features0, valid_idx0 = results[0]
    features1, valid_idx1 = results[1]

    # 拼接特征
    all_features = np.vstack([features0, features1])
    all_valid_idx = valid_idx0 + [idx + mid for idx in valid_idx1]

    print(f"✅ 双 GPU 提取完成! 总特征: {all_features.shape}")

    return all_features, all_valid_idx


if __name__ == "__main__":
    # 测试
    print("测试双 GPU 并行提取...")

    # 示例配置
    config = {
        'model_type': 'mpnn',
        'hidden_dim': 128,
        'num_layers': 2,
        'output_dim': 128,
        'dropout': 0.1,
        'pooling': 'mean',
        'batch_size': 32,
        'chunk_size': 512,
        'num_workers': 4,
        'add_hs': True,
        'seed': 42,
    }

    # 示例 SMILES
    test_smiles = ["CCO", "CC(C)O", "c1ccccc1"] * 100

    features, valid_idx = extract_gnn_features_dual_gpu(test_smiles, config)

    if features is not None:
        print(f"提取成功! 特征形状: {features.shape}")
