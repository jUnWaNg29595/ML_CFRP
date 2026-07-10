# CFRP智能预测平台 - 开发指南

## 项目架构

### 核心模块说明

#### 1. 虚拟筛选模块 (`core/virtual_screening.py`)

**主要功能:**
- 候选分子库生成与组合
- 化学规则过滤(环氧官能度、分子量等)
- 可合成性评估
- 配方可行性分析
- 多模型预测集成

**关键类和函数:**
```python
# 候选池生成
generate_candidate_pool(resin_smiles, hardener_smiles, mode='cartesian')

# 化学规则过滤
filter_candidates_by_epoxy_rules(df, rules=DEFAULT_EPOXY_RULES)

# 特征矩阵构建
build_feature_matrix(smiles_list, config)

# 预测与不确定度
predict_with_uncertainty(model, X, n_samples=100)
```

**默认化学规则:**
```python
DEFAULT_EPOXY_RULES = {
    "resin": {
        "min_epoxide": 1,           # 最小环氧官能度
        "max_epoxide": 4,           # 最大环氧官能度
        "min_aromatic_rings": 1,    # 最小芳香环数
        "min_mw": 180.0,            # 最小分子量
        "max_mw": 2000.0,           # 最大分子量
    },
    "hardener": {
        "min_mw": 60.0,
        "max_mw": 1000.0,
        "allowed_classes": ["amine", "anhydride", "phenol", "thiol", "imidazole"],
    },
}
```

#### 2. 分子特征模块 (`core/molecular_features.py`)

**支持的指纹类型:**
- Morgan (ECFP): 半径2, 2048位
- MACCS: 166位
- RDKit: 2048位
- Atom Pair: 2048位

**描述符类型:**
- RDKit: 210+描述符
- Mordred: 1800+描述符
- 3D描述符: 需要RDKit 3D坐标生成

**特征流程配置:**
```json
{
  "smiles_col": "SMILES",
  "hardener_col": "hardener_smiles",
  "feature_types": ["morgan", "maccs", "rdkit"],
  "morgan_radius": 2,
  "morgan_bits": 2048,
  "use_3d": false
}
```

#### 3. 模型训练模块 (`core/model_trainer.py`)

**支持的模型:**
- XGBoost: 默认模型,快速高效
- BNN: 贝叶斯神经网络,提供不确定度
- PINN: 物理信息神经网络
- TabNet: 深度表格学习
- GNN: 图神经网络(需要torch_geometric)
- Transformer系列: 需要大量数据
- AutoGluon: 自动化集成

**训练流程:**
```python
from core.model_trainer import train_model

result = train_model(
    X_train, y_train,
    model_type="XGBoost",
    params={
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
    },
    validation_data=(X_test, y_test),
)
```

#### 4. 模型解释模块 (`core/model_interpreter.py`)

**SHAP分析:**
```python
from core.model_interpreter import compute_shap_values

shap_values = compute_shap_values(
    model, X_train,
    background_samples=100,
    plot_type="summary"
)
```

**可视化类型:**
- Summary plot: 特征重要性排序
- Dependence plot: 特征依赖关系
- Interaction plot: 特征交互
- Force plot: 单样本解释

---

## 开发规范

### 代码风格

- Python 3.10+
- 遵循PEP 8
- 使用类型提示
- 函数文档字符串

### Git提交规范

```
<type>(<scope>): <subject>

<body>

<footer>
```

**类型:**
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式
- `refactor`: 重构
- `test`: 测试
- `chore`: 构建/工具

**示例:**
```
feat(virtual_screening): add PubChem candidate integration

- Add PubChem search by keywords
- Integrate with existing candidate pool
- Add caching for API responses

Closes #123
```

### 分支管理

```
main         # 主分支,稳定版本
develop      # 开发分支
feature/*    # 新功能分支
bugfix/*     # bug修复分支
release/*    # 发布分支
```

---

## 性能优化

### 1. 特征提取优化

```python
# 启用缓存
from core.molecular_features import FeatureCache
cache = FeatureCache(cache_dir=".cache/features")

# 并行化
features = extract_features(
    smiles_list,
    n_jobs=8,  # 使用8个进程
    batch_size=1000
)
```

### 2. 模型训练优化

**XGBoost参数:**
```python
params = {
    "tree_method": "hist",      # 使用hist算法,更快
    "n_jobs": 8,                # 并行化
    "early_stopping_rounds": 50, # 早停
}
```

**GPU加速:**
```python
# XGBoost GPU
params["tree_method"] = "gpu_hist"

# PyTorch GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

### 3. 虚拟筛选优化

```python
# 批量预测
predictions = model.predict(X, batch_size=10000)

# 缓存中间结果
cache_key = f"features_{hash(tuple(smiles_list))}"
```

---

## 常见问题

### Q1: 如何添加新的分子指纹?

在`core/molecular_features.py`中添加:

```python
def _compute_custom_fingerprint(mol, **kwargs):
    from rdkit.Chem import AllChem
    return AllChem.GetCustomFingerprint(mol, **kwargs)

# 在FEATURE_EXTRACTORS中注册
FEATURE_EXTRACTORS["custom"] = _compute_custom_fingerprint
```

### Q2: 如何添加新的模型?

在`core/model_trainer.py`中添加:

```python
def train_custom_model(X_train, y_train, params):
    from custom_library import CustomModel
    model = CustomModel(**params)
    model.fit(X_train, y_train)
    return model

# 在MODEL_REGISTRY中注册
MODEL_REGISTRY["CustomModel"] = {
    "train_fn": train_custom_model,
    "default_params": {...},
}
```

### Q3: 如何自定义化学规则?

```python
from core.virtual_screening import filter_candidates_by_epoxy_rules

custom_rules = {
    "resin": {
        "min_epoxide": 2,  # 更严格
        "min_mw": 300.0,   # 更高分子量
    },
}

filtered_df = filter_candidates_by_epoxy_rules(
    candidates_df,
    rules=custom_rules
)
```

---

## 测试

### 单元测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_virtual_screening.py

# 覆盖率报告
pytest --cov=core tests/
```

### 集成测试

```bash
# 测试完整流程
python scripts/test_full_pipeline.py
```

---

## 部署

### Docker部署

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

```bash
docker build -t cfrp-app .
docker run -p 8501:8501 cfrp-app
```

### 服务器部署

```bash
# 使用gunicorn + nginx
gunicorn --workers 4 --bind 0.0.0.0:8000 app:app

# nginx配置
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
    }
}
```

---

## 贡献指南

1. Fork本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

---

## 更新日志

### v1.4.5 (2025-01)
- 新增OpenMP线程优化
- 改进虚拟筛选性能
- 添加PubChem候选集成

### v1.4.4
- 新增配方可行性分析
- 改进化学规则过滤
- 修复特征提取bug

### v1.4.3
- 新增适用域分析
- 改进模型解释模块
- 优化UI布局

---

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件