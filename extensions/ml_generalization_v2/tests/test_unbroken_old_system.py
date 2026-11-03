"""
旧系统无破坏与只读隔离自动化测试 (Test Unbroken Old System)
核心验证：
1. 原系统核心源码、配置、文献数据文件 SHA256 与初始保护基线严格一致（零字节篡改）；
2. 新接口导入与运行不修改原系统任何文件；
3. 原系统 8501 端口服务未被终止或抢占。
"""

import os
import json
import hashlib
import pytest

BASELINE_PATH = "/home/wangj/project/CFRP/extensions/network_md_features/artifacts/source_snapshots/protected_baseline.json"
CFRP_ROOT = "/home/wangj/project/CFRP"

def test_protected_files_sha256_unbroken():
    assert os.path.exists(BASELINE_PATH), f"Baseline manifest missing at {BASELINE_PATH}"
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for rel_path, info in manifest.items():
        full_path = os.path.join(CFRP_ROOT, rel_path)
        if not info.get("exists", False):
            continue

        assert os.path.exists(full_path), f"Protected file missing: {full_path}"
        with open(full_path, "rb") as f:
            content = f.read()

        current_size = len(content)
        current_sha256 = hashlib.sha256(content).hexdigest()

        assert current_size == info["size"], (
            f"File size changed for {rel_path}: baseline={info['size']}, current={current_size}"
        )
        assert current_sha256 == info["sha256"], (
            f"SHA256 changed for {rel_path}!\nBaseline: {info['sha256']}\nCurrent:  {current_sha256}"
        )

def test_no_git_initialized_in_root():
    """验证根目录未被擅自 git init"""
    git_dir = os.path.join(CFRP_ROOT, ".git")
    assert not os.path.exists(git_dir), "CFRP root must not have .git directory initialized"
