# -*- coding: utf-8 -*-
"""一次性脚本：直连 hf-mirror 下载 ChemBERTa 到本地 HF 缓存。"""
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['NO_PROXY'] = 'hf-mirror.com,.hf-mirror.com'
os.environ['no_proxy'] = 'hf-mirror.com,.hf-mirror.com'

from huggingface_hub import snapshot_download

path = snapshot_download('seyonec/ChemBERTa-zinc-base-v1')
print('DOWNLOAD_OK:', path)
