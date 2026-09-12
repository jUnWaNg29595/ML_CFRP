# -*- coding: utf-8 -*-
"""专项验证：各种"无真实列名"路径下 SHAP 特征名恢复能力（对应线上 TabPFN 问题）"""
import sys, io, ast, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

np.random.seed(0)

N_REAL = 137          # 训练后实际特征数（145 canonical 被剔除 8 个 all-NaN 列）
N_SESSION = 145       # session_state.feature_cols 里的 canonical 数量
real_names = [f"MolDesc_{i:03d}" for i in range(N_REAL)]
session_names = [f"Canon_{i:03d}" for i in range(N_SESSION)]

# ---------- 1) 从 app.py 提取 _coerce_feature_frame / _build_split_snapshot_tables ----------
app_src = open("app.py", encoding="utf-8").read()
tree = ast.parse(app_src)
ns = {"pd": pd, "np": np, "re": re}
wanted = {"_coerce_feature_frame", "_coerce_target_array", "_build_split_snapshot_tables"}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in wanted:
        mod = ast.Module(body=[node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(mod), "app_extract", "exec"), ns)
assert "_coerce_feature_frame" in ns and "_build_split_snapshot_tables" in ns
print("[1] app.py 函数提取成功")

# ---------- 2) Case C: 快照保存（ndarray X_train_raw + 长度不匹配的 feature_cols） ----------
X_train_df = pd.DataFrame(np.random.randn(50, N_REAL), columns=real_names)
X_train_raw_arr = X_train_df.values  # ndarray，无列名
tables = ns["_build_split_snapshot_tables"](
    X_train_df, X_train_df.iloc[:10].copy(),
    np.random.randn(50), np.random.randn(10),
    feature_cols=session_names,  # 145 != 137 → 旧代码会写 Feature_i
    target_col="target",
    X_train_raw=X_train_raw_arr,
    X_test_raw=X_train_df.iloc[:10].values,
)
cols = list(tables["split_X_train"].columns)
assert cols[:3] == real_names[:3] and not any(c.startswith("Feature_") for c in cols), cols[:5]
print(f"[2] Case C 通过: 快照列名为真实名（旧代码此处会生成 Feature_i）: {cols[:3]}...")

# ---------- 3) Case D: 从带 Feature_i 表头的旧快照 CSV 恢复 ----------
legacy = pd.DataFrame(np.random.randn(20, N_REAL), columns=[f"Feature_{i}" for i in range(N_REAL)])
fixed = ns["_coerce_feature_frame"](legacy, session_names)          # 长度不匹配 → 保持占位
fixed2 = ns["_coerce_feature_frame"](legacy, real_names)            # 长度匹配 + 占位表头 → 覆盖为真实名
assert list(fixed.columns)[:2] == ["Feature_0", "Feature_1"]        # 不匹配时不覆盖
assert list(fixed2.columns)[:2] == real_names[:2]
print("[3] Case D 通过: 旧 Feature_i 快照在拿到正确长度名单时可自愈")

# 真实列名不被错误覆写
real_df = pd.DataFrame(np.random.randn(10, N_REAL), columns=real_names)
keep = ns["_coerce_feature_frame"](real_df, session_names)          # 145 != 137
assert list(keep.columns)[:2] == real_names[:2]
print("[4] 真实列名不会被长度不匹配列表覆写")

# ---------- 4) Case A/B: 解释器解析（ndarray / 占位列 DataFrame + 模型 feature_names_in_） ----------
from core.model_interpreter import resolve_feature_names_for_matrix, EnhancedModelInterpreter, _is_placeholder_feature_name

class FakeTabPFN:
    """模拟线上已训练 TabPFN：feature_names_in_ 存在但为 None"""
    feature_names_in_ = None
    def predict(self, X):
        return np.zeros(len(X))

mdl = FakeTabPFN()

# A1: ndarray + 错误长度名单 + 模型无名字 → 只能占位（位置正确）
resolved = resolve_feature_names_for_matrix(
    X_train_raw_arr, feature_names=session_names, model=mdl,
)
assert all(_is_placeholder_feature_name(n) for n in resolved) and len(resolved) == N_REAL
print("[5] A1 占位回退位置正确（模型无名字时）")

# A2: ndarray + fallback_feature_names（train_result['feature_names']）→ 解释器恢复真实名
interp = EnhancedModelInterpreter(
    mdl, X_train_df, pd.Series(np.random.randn(50)), X_train_df.iloc[:10].copy(), pd.Series(np.random.randn(10)),
    "TabPFN", feature_names=session_names,
    fallback_feature_names=real_names,
    max_samples=5,
)
assert list(interp.feature_names)[:3] == real_names[:3]
print(f"[6] A2 通过: ndarray 输入 + fallback 名单 → 恢复真实特征名 {list(interp.feature_names)[:3]}")

# B: 占位列 DataFrame + 模型注入真实 feature_names_in_ 后解析
mdl2 = FakeTabPFN()
mdl2.feature_names_in_ = np.asarray(real_names, dtype=object)   # 修复后 trainer 注入的效果
placeholder_df = pd.DataFrame(X_train_raw_arr, columns=[f"Feature_{i}" for i in range(N_REAL)])
resolved_b = resolve_feature_names_for_matrix(placeholder_df, feature_names=session_names, model=mdl2)
assert list(resolved_b)[:3] == real_names[:3]
print("[7] B 通过: 模型携带 feature_names_in_ 后，占位列矩阵也能解析出真实名")

# C2: pipeline 传入 resolver
class FakePipe:
    def __init__(self, names):
        self.feature_names_in_ = np.asarray(names, dtype=object)
resolved_c = resolve_feature_names_for_matrix(
    X_train_raw_arr, feature_names=session_names, model=FakeTabPFN(), pipeline=FakePipe(real_names),
)
assert list(resolved_c)[:3] == real_names[:3]
print("[8] C2 通过: pipeline.feature_names_in_ 也可作为名字来源")

print("\nALL PASSED ✓")
