# CFRP系统环境管理指南

## 问题说明

由于不同深度学习模型对PyTorch版本的要求不同，导致依赖冲突：

- **FT-Transformer** 需要 `torch<2.0`
- **PINN、AutoGluon、ANN** 需要 `torch>=2.6`
- **TabNet** 两个版本都支持

## 解决方案

### 方案1：单环境 + 模型替代（推荐）⭐

**保持当前Torch 2.x环境，使用TabNet替代FT-Transformer**

优势：
- ✅ 支持最多模型（PINN、AutoGluon、TabNet、ANN、树模型）
- ✅ 无需切换环境
- ✅ TabNet性能与FT-Transformer相当
- ✅ 维护简单

劣势：
- ❌ 无法使用FT-Transformer

**适用场景**：日常使用、追求稳定性

---

### 方案2：双环境策略

**创建两个独立的conda环境**

#### 主环境（CFRP_env）- Torch 2.x
```bash
conda activate CFRP_env
# 支持：PINN、AutoGluon、TabNet、ANN、树模型
```

#### 辅助环境（ft_transformer_env）- Torch 1.x
```bash
# 创建新环境
conda create -n CFRP1_env python=3.10
conda activate CFRP1_env

# 安装依赖
pip install torch==1.13.1
pip install rtdl pytorch-tabnet
pip install scikit-learn pandas numpy xgboost lightgbm catboost

# 支持：FT-Transformer、TabNet、树模型
```

**使用方法**：
1. 日常工作使用主环境（CFRP_env）
2. 需要FT-Transformer时切换到辅助环境
3. 在辅助环境中运行系统

优势：
- ✅ 可以使用所有模型
- ✅ 环境隔离，互不影响

劣势：
- ❌ 需要手动切换环境
- ❌ 维护两套环境

**适用场景**：确实需要FT-Transformer的场景

---

### 方案3：快速环境切换

**使用提供的脚本快速切换**

#### Windows用户
```bash
# 双击运行
switch_environment.bat

# 或命令行运行
cd c:\Users\wangj\Desktop\CFRP系统
switch_environment.bat
```

#### 查看当前环境状态
```bash
python environment_helper.py
```

---

## 环境对比

| 环境 | Torch版本 | 支持模型 | 不支持模型 |
|------|-----------|----------|------------|
| **Torch 2.x（推荐）** | 2.6.0+ | PINN, AutoGluon, TabNet, ANN, 树模型 | FT-Transformer |
| **Torch 1.x** | 1.13.1 | FT-Transformer, TabNet, 树模型 | PINN, AutoGluon, ANN |
| **最小环境** | 无 | 树模型（XGBoost等） | 所有深度学习模型 |

---

## 模型性能对比

针对7000×37表格数据的预期性能：

| 模型 | 预期R² | 训练速度 | 可解释性 | 推荐度 |
|------|--------|----------|----------|--------|
| **XGBoost** | 0.80 | ⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **TabNet** | 0.80-0.85 | ⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **LightGBM** | 0.80-0.82 | ⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **FT-Transformer** | 0.80-0.85 | ⚡ | ⭐⭐ | ⭐⭐⭐ |
| **PINN** | 0.71-0.74 | ⚡⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

**结论**：TabNet可以完美替代FT-Transformer，且更易用。

---

## 推荐配置

### 对于大多数用户
```
环境：Torch 2.x（当前环境）
主力模型：XGBoost、TabNet、LightGBM
备选模型：PINN（需要物理可解释性时）
```

### 对于必须使用FT-Transformer的用户
```
方案：双环境策略
主环境：Torch 2.x（日常使用）
辅助环境：Torch 1.x（FT-Transformer专用）
```

---

## 常见问题

### Q1: 为什么不能同时支持所有模型？
A: FT-Transformer依赖的rtdl库要求torch<2.0，而PINN、AutoGluon需要torch>=2.6，两者冲突无法共存。

### Q2: TabNet真的能替代FT-Transformer吗？
A: 是的。在表格数据上，TabNet和FT-Transformer性能相当，且TabNet：
- 训练速度更快
- 可解释性更强（attention机制）
- 社区支持更好
- 无依赖冲突

### Q3: 如何快速切换环境？
A: 运行 `switch_environment.bat`（Windows）或使用conda切换环境。

### Q4: 切换环境会丢失数据吗？
A: 不会。数据和模型文件独立于Python环境，切换环境只影响可用的模型类型。

---

## 文件说明

- `environment_manager.py` - 环境管理器，自动检测模型可用性
- `environment_profiles.py` - 环境配置文件定义
- `environment_helper.py` - 环境状态查看工具
- `switch_environment.bat` - Windows环境切换脚本

---

## 联系支持

如有问题，请查看系统日志或联系技术支持。
