"""模型 artifact 反序列化必须能在不同 CUDA 拓扑的机器之间迁移，且**不得破坏设备一致性**。

两个用户实测报错（同一根因链）
------------------------------
报错 1（加载失败）::

    模型预览失败: Attempting to deserialize object on CUDA device 1 but
    torch.cuda.device_count() is 1. Please use torch.load with map_location
    to map your storages to an existing device.

报错 2（设备不一致，由"一律 map to cpu"的错误修法引入）::

    【Tg.joblib】预测失败: Expected all tensors to be on the same device, but
    found at least two devices, cpu and cuda:0! (when checking argument for
    argument mat1 in method wrapper_CUDA_addmm)

根因
----
模型在 ``cuda:N`` 上训练并保存，artifact 里 pickle 了绑定该设备的 tensor，
以及 ``self.device = 'cuda:N'`` 这样的**普通字符串属性**（map_location 改不了它）。

- 在缺少该设备的机器上：PyTorch 的 ``torch.storage._load_from_bytes`` 内部调用
  ``torch.load(io.BytesIO(b), weights_only=False)``，未传 map_location → 加载失败。
- 若把所有 tensor 一律映射到 CPU：在**有**该设备的机器上，权重落到 CPU 而
  ``self.device`` 仍是 ``cuda:0``，forward 里 ``x.to(self.device)`` 把输入搬到
  cuda:0 → 设备不一致（报错 2）。

正确修法
--------
``map_location`` 用**可调用对象**（``core.model_io._portable_map_location``）：

1. 目标设备在当前机器可用 → ``storage.cuda(index)`` 原地恢复（与训练时一致）；
2. 目标设备不可用 → 原样返回 storage（它在 CPU 上重建，即留在 CPU）。

注意：可调用版 map_location 必须返回 **storage 对象**（不是设备字符串），
torch 的 legacy/zip 两条路径都如此。

随后 :func:`core.model_io._repair_loaded_object_devices` 把指向**不可用** CUDA
设备的 ``device`` 字符串属性同步修正为 ``cpu``，使 forward 不会设备不一致。
"""

from __future__ import annotations

import io
import json

import pytest

torch = pytest.importorskip("torch")
joblib = pytest.importorskip("joblib")

import core.model_io as model_io
from core.model_io import dumps_artifact, loads_artifact


def _cuda_tensor_on_device(index: int):
    """在指定 CUDA 设备上造一个 tensor；设备不存在时跳过。"""
    if not torch.cuda.is_available() or torch.cuda.device_count() <= index:
        pytest.skip(f"需要至少 {index + 1} 块 GPU")
    return torch.randn(4, device=f"cuda:{index}")


class _DeviceAwareNet(torch.nn.Module):
    """带 ``self.device`` 字符串属性的模型（最常见的自建 NN 写法）。"""

    def __init__(self, device_attr: str):
        super().__init__()
        self.device = device_attr
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x.to(self.device))


def _artifact_with(tensor):
    return {
        "artifact_version": "1.0",
        "model_name": "CudaSavedModel",
        "target_col": "tg_c",
        "feature_cols": ["a", "b"],
        "model": {"weights": tensor},
        "extra": {},
    }


# ---------------------------------------------------------------------------
# 报错 1：缺少目标设备时必须能加载
# ---------------------------------------------------------------------------

def test_artifact_saved_on_cuda1_loads_when_only_one_gpu(monkeypatch):
    """核心回归：CUDA:1 保存的 artifact 在单卡机器上必须能加载。

    模拟用户环境：cuda:1 不可用（device_count=1 且不可用）。
    """
    data = dumps_artifact(_artifact_with(_cuda_tensor_on_device(1)))

    monkeypatch.setattr(model_io, "_cuda_device_usable", lambda index: False)

    artifact = loads_artifact(data)

    assert artifact["model_name"] == "CudaSavedModel"
    # 不可用设备 → tensor 落回 CPU
    assert artifact["model"]["weights"].device.type == "cpu"


def test_stale_device_attribute_repaired_for_fallback(monkeypatch):
    """落回 CPU 后，指向不可用设备的 ``device`` 属性必须被同步修正为 cpu。

    若不修正，forward 里 ``x.to(self.device)`` 会把输入搬到不存在的设备，
    报设备不一致错误。
    """
    net = _DeviceAwareNet("cuda:1").cuda(1)
    data = dumps_artifact(
        {"artifact_version": "1.0", "model_name": "StaleDevice", "model": net}
    )

    monkeypatch.setattr(model_io, "_cuda_device_usable", lambda index: False)

    restored = loads_artifact(data)["model"]

    assert restored.device == "cpu"
    assert next(restored.parameters()).device.type == "cpu"


def test_fallback_model_forward_works(monkeypatch):
    """端到端：不可用设备的模型加载后 forward 必须成功且数值一致。"""
    torch.manual_seed(0)
    net = _DeviceAwareNet("cuda:1").cuda(1)
    x = torch.randn(2, 4)
    with torch.no_grad():
        expected = net(x).cpu().numpy()
    data = dumps_artifact(
        {"artifact_version": "1.0", "model_name": "StaleDevice", "model": net}
    )

    monkeypatch.setattr(model_io, "_cuda_device_usable", lambda index: False)

    restored = loads_artifact(data)["model"]

    with torch.no_grad():
        got = restored(x).cpu().numpy()
    assert abs(float((expected - got).max())) < 1e-6


# ---------------------------------------------------------------------------
# 报错 2：设备可用时必须原地恢复（不得破坏设备一致性）
# ---------------------------------------------------------------------------

def test_device0_model_stays_on_device0_and_forward_works(monkeypatch):
    """核心回归（报错 2）：cuda:0 保存的模型在本机必须保持 cuda:0 且 forward 正常。

    早期"一律 map to cpu"的修法会把权重改到 CPU，而 ``self.device`` 仍是
    'cuda:0'，导致 ``Expected all tensors to be on the same device, but found
    at least two devices, cpu and cuda:0!``。本测试锁定正确行为。
    """
    torch.manual_seed(3)
    net = _DeviceAwareNet("cuda:0").cuda(0)
    x = torch.randn(2, 4)
    with torch.no_grad():
        expected = net(x).cpu().numpy()
    data = dumps_artifact(
        {"artifact_version": "1.0", "model_name": "Cuda0Model", "model": net}
    )

    restored = loads_artifact(data)["model"]

    # 权重保持在 cuda:0，device 属性不变 → 前后端一致
    assert next(restored.parameters()).device.type == "cuda"
    assert restored.device == "cuda:0"
    with torch.no_grad():
        got = restored(x).cpu().numpy()
    assert abs(float((expected - got).max())) < 1e-6


def test_device0_stays_even_if_other_device_missing(monkeypatch):
    """cuda:0 可用时不得被牵连：即使 cuda:1 不可用，cuda:0 的模型也保持原位。

    这锁定"按设备逐一判断"而不是"看到 CUDA 就全部转 CPU"。
    """
    tensor = _cuda_tensor_on_device(0)
    data = dumps_artifact(_artifact_with(tensor))

    monkeypatch.setattr(
        model_io, "_cuda_device_usable", lambda index: index == 0
    )

    artifact = loads_artifact(data)

    assert artifact["model"]["weights"].device.type == "cuda"
    assert artifact["model"]["weights"].device.index == 0


# ---------------------------------------------------------------------------
# 通用回归
# ---------------------------------------------------------------------------

def test_artifact_saved_on_cuda1_loads_normally_too():
    """双卡机器上 cuda:1 保存的模型保持原位（不得因修复而改变行为）。"""
    tensor = _cuda_tensor_on_device(1)
    data = dumps_artifact(_artifact_with(tensor))

    artifact = loads_artifact(data)

    assert artifact["model"]["weights"].device.type == "cuda"
    assert artifact["model"]["weights"].device.index == 1


def test_cpu_only_artifact_unaffected():
    """纯 CPU artifact 的加载行为完全不变。"""
    artifact = {
        "artifact_version": "1.0",
        "model_name": "CpuModel",
        "target_col": "tg_c",
        "feature_cols": ["a"],
        "model": {"weights": torch.randn(3)},
        "extra": {},
    }
    restored = loads_artifact(dumps_artifact(artifact))

    assert restored["model_name"] == "CpuModel"
    assert restored["model"]["weights"].device.type == "cpu"


def test_load_does_not_leak_patched_loader():
    """修复用的 monkeypatch 必须是临时的，不得污染全局 torch 状态。"""
    import torch.storage as ts

    original = ts._load_from_bytes
    loads_artifact(dumps_artifact(_artifact_with(torch.randn(2))))

    assert ts._load_from_bytes is original


def test_cpu_fallback_preserves_tensor_values(monkeypatch):
    """落回 CPU 后数值必须完全一致（不得被改写）。"""
    tensor = _cuda_tensor_on_device(1)
    expected = tensor.detach().cpu().numpy().tolist()
    data = dumps_artifact(_artifact_with(tensor))

    monkeypatch.setattr(model_io, "_cuda_device_usable", lambda index: False)
    restored = loads_artifact(data)

    assert restored["model"]["weights"].detach().cpu().numpy().tolist() == expected


def test_artifact_without_artifact_version_still_wraps():
    """既有兼容逻辑不受影响：裸 pipeline 仍被包装。"""
    import numpy as np
    from sklearn.linear_model import LinearRegression

    model = LinearRegression().fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))
    restored = loads_artifact(dumps_artifact(model))

    assert "model" in restored or "pipeline" in restored


def test_preview_artifact_handles_cuda1_saved_model(monkeypatch):
    """UI「模型预览」路径（load_model_artifact_bytes）同样不得失败。"""
    from UserPrediction import preview_artifact

    data = dumps_artifact(_artifact_with(_cuda_tensor_on_device(1)))
    monkeypatch.setattr(model_io, "_cuda_device_usable", lambda index: False)

    artifact, preview = preview_artifact(data)

    assert preview["model_name"] == "CudaSavedModel"
    assert json.dumps(preview)  # 可 JSON 序列化（UI 要 st.json）


def test_external_augmenter_path_loads_cuda0_model():
    """模型补齐数据路径（external_feature_augmenter → loads_artifact）不得设备不一致。

    该路径把 artifact['pipeline'] 或 ['model'] 直接拿去 predict；若权重被错误
    改到 CPU 而 self.device 仍是 cuda:0，predict 就报错 2。
    """
    torch.manual_seed(5)
    net = _DeviceAwareNet("cuda:0").cuda(0)
    data = dumps_artifact(
        {"artifact_version": "1.0", "model_name": "Tg", "model": net}
    )

    restored = loads_artifact(data)["model"]
    x = torch.randn(3, 4)

    with torch.no_grad():
        restored(x)  # 不得抛设备不一致
