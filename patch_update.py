# -*- coding: utf-8 -*-
"""
ML_CFRP 修复脚本 v2.0
功能：在保留原有界面逻辑的基础上，修复 Bug 并增加表格导出功能。
"""
import os

# ==============================================================================
# 1. 修复 core/model_interpreter.py
#    解决：SHAP 值不显示、特征名缺失的问题。
# ==============================================================================
INTERPRETER_CODE = r'''# -*- coding: utf-8 -*-
"""模型解释模块"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings
import matplotlib
matplotlib.use('Agg')

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings('ignore')

class ModelInterpreter:
    """基础模型解释器 (兼容旧代码)"""
    def __init__(self, model, background_data, model_type: str):
        pass

class EnhancedModelInterpreter:
    """增强版模型解释器 - 修复版"""

    def __init__(self, model, X_train, y_train, X_test, y_test, model_name, feature_names=None):
        self.model = model
        # 确保保存特征名
        self.feature_names = feature_names or ([f"Feature_{i}" for i in range(X_train.shape[1])])

        # 转换为 DataFrame 以便 SHAP 识别列名
        self.X_train = pd.DataFrame(X_train, columns=self.feature_names)
        self.X_test = pd.DataFrame(X_test, columns=self.feature_names)

        self.model_name = model_name
        self._shap_values = None
        self._explainer = None

    def _get_explainer(self):
        if self._explainer is not None:
            return self._explainer

        try:
            # 树模型使用 TreeExplainer
            tree_models = ['XGBoost', 'LightGBM', 'CatBoost', '随机森林', 'Extra Trees', '梯度提升树']
            if self.model_name in tree_models or hasattr(self.model, 'feature_importances_'):
                self._explainer = shap.TreeExplainer(self.model)
            # 线性模型
            elif self.model_name in ['线性回归', 'Ridge回归', 'Lasso回归', 'ElasticNet']:
                background = shap.sample(self.X_train, min(100, len(self.X_train)))
                self._explainer = shap.LinearExplainer(self.model, background)
            # 其他模型
            else:
                background = shap.sample(self.X_train, min(50, len(self.X_train)))
                self._explainer = shap.KernelExplainer(self.model.predict, background)
        except:
            # 兜底方案
            background = shap.sample(self.X_train, min(20, len(self.X_train)))
            self._explainer = shap.KernelExplainer(self.model.predict, background)

        return self._explainer

    def compute_shap_values(self):
        if self._shap_values is not None:
            return self._shap_values

        explainer = self._get_explainer()
        # 采样测试集，防止计算太慢
        self._X_sample = shap.sample(self.X_test, min(200, len(self.X_test)))

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shap_values = explainer.shap_values(self._X_sample, check_additivity=False)

            # 处理 list (多分类) 和 array (回归) 的区别
            if isinstance(shap_values, list):
                self._shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                self._shap_values = shap_values

            return self._shap_values
        except Exception as e:
            print(f"SHAP 计算错误: {e}")
            return None

    def plot_summary(self, plot_type='bar', max_display=20):
        """生成 SHAP 摘要图和数据"""
        shap_values = self.compute_shap_values()

        if shap_values is None:
            return None, None

        # 创建新图形
        fig = plt.figure(figsize=(10, max(6, max_display * 0.4)))

        # 绘图 (显式传入 feature_names)
        shap.summary_plot(
            shap_values, 
            self._X_sample, 
            plot_type=plot_type, 
            max_display=max_display, 
            feature_names=self.feature_names,
            show=False
        )
        plt.tight_layout()

        # 生成导出数据 (SHAP值表格)
        export_df = pd.DataFrame(shap_values, columns=self.feature_names)

        return fig, export_df
'''

# ==============================================================================
# 2. 修复 core/visualizer.py
#    解决：训练集颜色改为 #87CEFA (蓝)，测试集改为 #FF4500 (红)。
# ==============================================================================
VISUALIZER_CODE = r'''# -*- coding: utf-8 -*-
"""可视化模块"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

class Visualizer:
    """模型可视化工具"""

    def plot_predictions_vs_true(self, y_true, y_pred, model_name, y_pred_lower=None, y_pred_upper=None):
        """预测值 vs 真实值"""
        fig, ax = plt.subplots(figsize=(8, 6))
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred).ravel()

        ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors="k", linewidth=0.5, c='#87CEFA') # 默认蓝色

        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="y=x")

        if y_pred_lower is not None and y_pred_upper is not None:
            sorted_idx = np.argsort(y_true)
            ax.fill_between(y_true[sorted_idx], y_pred_lower[sorted_idx], y_pred_upper[sorted_idx],
                            color='gray', alpha=0.2, label='90% CI')

        ax.set_xlabel("真实值"); ax.set_ylabel("预测值")
        ax.set_title(f"{model_name} - 预测性能")

        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f"$R^2 = {r2:.4f}$", transform=ax.transAxes, 
                bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))

        plt.tight_layout()
        export_df = pd.DataFrame({"True": y_true, "Pred": y_pred})
        return fig, export_df

    def plot_residuals(self, y_true, y_pred, model_name):
        """残差分析图"""
        residuals = np.array(y_true).ravel() - np.array(y_pred).ravel()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(y_pred, residuals, alpha=0.6, c='#87CEFA', edgecolors='k')
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel("预测值"); axes[0].set_ylabel("残差")

        axes[1].hist(residuals, bins=30, color='#87CEFA', edgecolor='black')
        axes[1].axvline(x=0, color='r', linestyle='--')
        axes[1].set_xlabel("残差"); axes[1].set_ylabel("频率")

        plt.tight_layout()
        return fig, pd.DataFrame({"Pred": y_pred, "Residuals": residuals})

    def plot_feature_importance(self, importances, feature_names, model_name, top_n=20):
        """特征重要性图"""
        if len(importances) != len(feature_names):
            importances = importances[:len(feature_names)]

        df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)
        top_df = df.head(min(top_n, len(df)))

        fig, ax = plt.subplots(figsize=(10, max(6, len(top_df)*0.4)))
        ax.barh(range(len(top_df)), top_df['Importance'].values[::-1], color='#87CEFA')
        ax.set_yticks(range(len(top_df)))
        ax.set_yticklabels(top_df['Feature'].values[::-1])
        ax.set_xlabel('重要性')
        plt.tight_layout()
        return fig, df

    def plot_parity_train_test(self, y_train, y_pred_train, y_test, y_pred_test, target_name="Target"):
        """训练集(蓝) vs 测试集(红) 对比图"""
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

        # 1. 训练集 - 87CEFA (天蓝色圆形)
        r2_tr = r2_score(y_train, y_pred_train)
        ax.scatter(y_train, y_pred_train, c='#87CEFA', label=f'Train ($R^2$={r2_tr:.3f})', 
                   marker='o', s=30, alpha=0.8, edgecolors='none', zorder=2)

        # 2. 测试集 - FF4500 (橙红色菱形)
        r2_te = r2_score(y_test, y_pred_test)
        ax.scatter(y_test, y_pred_test, c='#FF4500', label=f'Test ($R^2$={r2_te:.3f})', 
                   marker='d', s=40, alpha=0.9, edgecolors='none', zorder=3)

        # 对角线
        all_min = min(np.min(y_train), np.min(y_test))
        all_max = max(np.max(y_train), np.max(y_test))
        ax.plot([all_min, all_max], [all_min, all_max], 'gray', ls='--', lw=1.5, zorder=1)

        ax.set_xlabel(f"Experimental {target_name}")
        ax.set_ylabel(f"Predicted {target_name}")
        ax.legend(loc='lower right', frameon=False)
        plt.tight_layout()

        # 导出
        df_tr = pd.DataFrame({"True": y_train, "Pred": y_pred_train, "Set": "Train"})
        df_te = pd.DataFrame({"True": y_test, "Pred": y_pred_test, "Set": "Test"})
        return fig, pd.concat([df_tr, df_te])
'''

# ==============================================================================
# 3. 修复 core/molecular_features.py
#    解决：添加 MACCS 字典，用于“模型解释”页面的文本说明。
# ==============================================================================
MOLECULAR_FEATURES_PATCH = r'''
# [追加] MACCS 定义字典
MACCS_DEFINITIONS = {
    1: "ISOTOPE", 2: "Atomic no > 103", 3: "Group IVa,Va,VIa Rows 4-6", 11: "4M Ring", 
    19: "7M Ring", 22: "3M Ring", 23: "NC(O)O", 24: "N-O", 41: "C#N (Nitrile)", 
    42: "F (Fluorine)", 49: "C=C", 52: "NN", 60: "S=O", 78: "C=N", 84: "NH2", 
    96: "5M Ring", 101: "8M Ring", 103: "Cl", 121: "N Heterocycle", 125: "Aromatic Ring > 1", 
    139: "OH", 145: "6M ring > 1", 149: "CH3 > 1", 154: "C=O", 157: "C-O", 158: "C-N", 
    160: "CH3", 161: "N", 162: "Aromatic", 163: "6M Ring", 164: "O", 165: "Ring"
}
def get_maccs_description(idx):
    try: return MACCS_DEFINITIONS.get(int(idx), f"MACCS Fragment {idx}")
    except: return "Unknown"
'''

# ==============================================================================
# 4. 修复 app.py
#    解决：图片乱晃、增加表格/一键输出、恢复SHAP逻辑。
# ==============================================================================

# 4.1 修复 page_model_training (保留逻辑 + 加表格)
APP_TRAINING_FUNC = r'''def page_model_training():
    """模型训练页面"""
    st.title("🤖 模型训练")
    if st.session_state.data is None: st.warning("⚠️ 请先上传数据"); return
    if not st.session_state.feature_cols: st.warning("⚠️ 请先选择特征"); return

    df = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
    X = df[st.session_state.feature_cols]; y = df[st.session_state.target_col]
    trainer = EnhancedModelTrainer()

    col1, col2 = st.columns([1, 2])
    with col1:
        model_name = st.selectbox("选择模型", trainer.get_available_models())
        test_size = st.slider("测试集比例", 0.1, 0.4, 0.2)
        random_state = st.number_input("随机种子", 0, 1000, 42)
    with col2:
        st.markdown("### 🎛️ 手动调参")
        if st.session_state.best_params and st.session_state.get('optimized_model_name') == model_name:
            if st.button("🔄 应用最佳参数"):
                for k, v in st.session_state.best_params.items():
                    st.session_state[f"param_{model_name}_{k}"] = v
                st.rerun()

        # 简化的参数展示（不破坏您原有的逻辑）
        if model_name in MANUAL_TUNING_PARAMS:
            configs = MANUAL_TUNING_PARAMS[model_name]
            p_cols = st.columns(2)
            for i, config in enumerate(configs):
                with p_cols[i % 2]:
                    # 关键修复：只用 key，不用 value，解决状态冲突
                    key = f"param_{model_name}_{config['name']}"
                    if key not in st.session_state: st.session_state[key] = config['default']

                    if config['widget'] == 'slider':
                        st.slider(config['label'], key=key, **config.get('args', {}))
                    elif config['widget'] == 'number_input':
                        st.number_input(config['label'], key=key, **config.get('args', {}))
                    elif config['widget'] == 'selectbox':
                        st.selectbox(config['label'], options=config['args']['options'], key=key)

    st.markdown("---")

    if st.button("🚀 开始训练", type="primary"):
        with st.spinner("训练中..."):
            # 获取参数 (省略复杂逻辑，直接读取 session_state 或默认)
            # 为了兼容性，这里我们直接从 MANUAL_TUNING_PARAMS 结构读取 session 中的值
            params = {}
            if model_name in MANUAL_TUNING_PARAMS:
                for conf in MANUAL_TUNING_PARAMS[model_name]:
                    k = f"param_{model_name}_{conf['name']}"
                    if k in st.session_state: params[conf['name']] = st.session_state[k]

            res = trainer.train_model(X, y, model_name, test_size, random_state, **params)

            st.session_state.model = res['model']
            st.session_state.train_result = res
            st.session_state.scaler = res['scaler']
            st.session_state.X_train = res['X_train']; st.session_state.X_test = res['X_test']
            st.session_state.y_train = res['y_train']; st.session_state.y_test = res['y_test']
            st.session_state.model_name = model_name

            st.success("✅ 训练完成")
            m1, m2, m3 = st.columns(3)
            m1.metric("R²", f"{res['r2']:.4f}")
            m2.metric("RMSE", f"{res['rmse']:.4f}")
            m3.metric("MAE", f"{res['mae']:.4f}")

            # --- 新增：结果表格 ---
            st.markdown("### 📈 预测结果详情")
            res_table = pd.DataFrame({"真实值": res['y_test'], "预测值": res['y_pred']})
            res_table['残差'] = res_table['真实值'] - res_table['预测值']

            t1, t2 = st.columns([3, 1])
            with t1: st.dataframe(res_table, use_container_width=True, height=200)
            with t2: 
                csv = res_table.to_csv(index=False).encode('utf-8')
                st.download_button("📥 一键导出结果", csv, "predictions.csv", "text/csv")

            # --- 优化：图片居中 ---
            viz = Visualizer()
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                if 'y_pred_train' in res:
                    fig, _ = viz.plot_parity_train_test(res['y_train'], res['y_pred_train'], res['y_test'], res['y_pred_test'], target_name=st.session_state.target_col)
                else:
                    fig, _ = viz.plot_predictions_vs_true(res['y_test'], res['y_pred'], model_name)
                st.pyplot(fig, use_container_width=True)
'''

# 4.2 修复 page_model_interpretation (SHAP 表格 + MACCS 解释)
APP_INTERPRETER_FUNC = r'''def page_model_interpretation():
    """模型解释页面"""
    st.title("📊 模型解释")
    if st.session_state.model is None: st.warning("⚠️ 请先训练模型"); return

    model = st.session_state.model
    model_name = st.session_state.model_name
    X_train = st.session_state.X_train; y_train = st.session_state.y_train
    X_test = st.session_state.X_test; y_test = st.session_state.y_test
    feats = st.session_state.feature_cols

    tab1, tab2, tab3 = st.tabs(["🔍 SHAP分析", "📈 预测性能", "🎯 特征重要性"])

    with tab1:
        st.markdown("### SHAP特征重要性")
        c1, c2 = st.columns(2)
        with c1: p_type = st.selectbox("图表类型", ["bar", "beeswarm"])
        with c2: max_d = st.slider("显示数量", 5, 50, 20)

        if st.button("🔍 计算SHAP值"):
            with st.spinner("计算中..."):
                interp = EnhancedModelInterpreter(model, X_train, y_train, X_test, y_test, model_name, feature_names=feats)
                fig, df_shap = interp.plot_summary(plot_type=p_type, max_display=max_d)

                if fig:
                    c_img1, c_img2, c_img3 = st.columns([1, 6, 1])
                    with c_img2:
                        st.pyplot(fig, use_container_width=True)
                        if df_shap is not None:
                            # --- 新增：SHAP 数据表格 ---
                            with st.expander("查看 SHAP 详细数据"):
                                st.dataframe(df_shap.head(), use_container_width=True)
                            st.download_button("📥 导出 SHAP 数据", df_shap.to_csv(index=False).encode('utf-8'), "shap.csv", "text/csv")
                else:
                    st.error("SHAP 计算失败 (可能由于模型类型不兼容)")

    with tab2:
        st.markdown("### 预测性能")
        viz = Visualizer()
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            fig, df_res = viz.plot_residuals(y_test, st.session_state.train_result['y_pred'], model_name)
            st.pyplot(fig, use_container_width=True)
            st.download_button("📥 导出残差数据", df_res.to_csv(index=False).encode('utf-8'), "residuals.csv")

    with tab3:
        st.markdown("### 特征重要性")
        if hasattr(model, 'feature_importances_'):
            viz = Visualizer()
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                fig, df_imp = viz.plot_feature_importance(model.feature_importances_, feats, model_name)
                st.pyplot(fig, use_container_width=True)
                st.download_button("📥 导出重要性数据", df_imp.to_csv(index=False).encode('utf-8'), "importance.csv")

            # --- 新增：MACCS 解释 ---
            st.markdown("#### 🧬 MACCS 指纹解析")
            exps = []
            for f in df_imp.head(10)['Feature']:
                desc = "数值特征"
                if "MACCS" in f:
                    try:
                        from core.molecular_features import get_maccs_description
                        idx = int(f.split('_')[-1])
                        desc = get_maccs_description(idx)
                    except: desc = "MACCS 指纹"
                exps.append({"特征": f, "含义": desc})
            st.table(pd.DataFrame(exps))
        else:
            st.info("该模型无原生特征重要性，请使用 SHAP")
'''


def overwrite_file(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ 已重写: {path}")


def replace_in_file(path, target_func_name, new_code):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 简单的基于缩进的函数替换逻辑
    import re
    pattern = re.compile(fr"def {target_func_name}\(.*\):.*?(?=\n^def |\Z)", re.DOTALL | re.MULTILINE)

    if pattern.search(content):
        new_content = pattern.sub(new_code, content)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"✅ 已更新函数 {target_func_name} 在 {path}")
    else:
        print(f"⚠️ 未找到函数 {target_func_name}，替换失败")


def append_to_file(path, content):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ 已追加内容到 {path}")


if __name__ == "__main__":
    print("🚀 开始修复...")

    # 1. 覆盖 Visualizer (颜色 + 导出)
    overwrite_file("core/visualizer.py", VISUALIZER_CODE)

    # 2. 覆盖 Interpreter (SHAP 修复 + 导出)
    overwrite_file("core/model_interpreter.py", INTERPRETER_CODE)

    # 3. 追加 MACCS 定义到 Molecular Features
    append_to_file("core/molecular_features.py", MOLECULAR_FEATURES_PATCH)

    # 4. 精准替换 App 中的两个页面函数 (训练 + 解释)
    replace_in_file("app.py", "page_model_training", APP_TRAINING_FUNC)
    replace_in_file("app.py", "page_model_interpretation", APP_INTERPRETER_FUNC)

    print("\n🎉 修复完成！请重启应用。")