# -*- coding: utf-8 -*-
"""一次性脚本：端到端冒烟测试——离线实例化 ChemBERTa 特征提取器。"""
import os
import sys

# 模拟真实运行：先应用项目的统一网络配置（含 NO_PROXY 白名单修复）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['HF_HUB_OFFLINE'] = '1'  # 强制纯离线，验证缓存完整性
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

from core.molecular_features import SmilesTransformerEmbeddingExtractor

ext = SmilesTransformerEmbeddingExtractor()
print('AVAILABLE:', ext.AVAILABLE)
print('hidden_size:', ext.hidden_size)

smiles_list = ['CC(=O)Oc1ccccc1C(=O)O', 'c1ccccc1', 'CN1C=NC2=C1C(=O)N(C)C(=O)N2C']
embs = ext.smiles_to_embeddings(smiles_list)
print('embedding shape:', getattr(embs, 'shape', None))
print('SMOKE_TEST_OK')
