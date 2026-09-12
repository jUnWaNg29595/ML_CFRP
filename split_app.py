# -*- coding: utf-8 -*-
"""
AST 驱动的 app.py 拆分生成器：
  1. app_lib.py      —— 全部 def/class/常量/幂等环境设置（每进程仅执行一次）
  2. app.py          —— 薄入口（ENTRY 段 + st.navigation 分组导航 + 尾部）
  3. app_pages/*.py  —— 19 个 st.Page 薄封装
  4. tests 的 import app → import app_lib as app
文本按原始行段精确切割（非重新排版），保证定义逐字节一致。
"""
import ast, os, shutil, sys, json
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

SRC = "app.py"
BAK = "app.py.pre_split.bak"
LIB = "app_lib.py"
PAGES_DIR = "app_pages"

# 幂等重跑：若入口已被覆盖为薄文件，则从备份读原始源码
if os.path.exists(BAK) and os.path.getsize(BAK) > 100000:
    src = open(BAK, encoding="utf-8").read()
else:
    src = open(SRC, encoding="utf-8").read()
lines = src.split("\n")
tree = ast.parse(src)

# ---------------------------------------------------------------- 分类器
ENTRY_FUNCS = {
    "init_session_state", "_get_or_create_session_id", "_maybe_auto_restore",
    "render_sidebar", "render_top_status_bar", "_render_global_task_lock",
    "_maybe_autosave_session", "_save_session_snapshot", "_save_session_snapshot_async",
}

def uses_st(node):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute):
            base = sub
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id == "st":
                return True
    return False

def classify(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return "LIB"
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return "LIB"
    if isinstance(node, ast.Expr):
        c = node.value
        if isinstance(c, ast.Call):
            fn = c.func
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == "st":
                return "ENTRY"
            if isinstance(fn, ast.Name) and fn.id in ENTRY_FUNCS:
                return "ENTRY"
            return "ENTRY" if uses_st(node) else "LIB"
        return "LIB"
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        return "ENTRY" if uses_st(node) else "LIB"
    if isinstance(node, ast.Try) or (hasattr(ast, "TryStar") and isinstance(node, ast.TryStar)):
        subs = list(node.body) + [s for h in node.handlers for s in h.body] + list(node.orelse) + list(node.finalbody)
        cats = {classify(s) for s in subs}
        if cats == {"LIB"}:
            return "LIB"
        if cats == {"ENTRY"}:
            return "ENTRY"
        return "REVIEW"
    if isinstance(node, ast.If):
        if uses_st(node.test):
            return "ENTRY"
        cats = {classify(s) for s in node.body + node.orelse}
        if cats == {"LIB"}:
            return "LIB"
        if cats == {"ENTRY"}:
            return "ENTRY"
        return "REVIEW"
    if isinstance(node, (ast.For, ast.While, ast.With)):
        return "REVIEW"
    return "REVIEW"

# 人工裁决的 REVIEW 段（依据逐段人工检查结论）
OVERRIDES = {
    109: "LIB",    # 环境变量守卫的性能补丁（幂等，无 st 渲染）
    192: "LIB",    # logging 配置循环
    201: "LIB",    # ScriptRunContext 日志补丁 try
    207: "LIB",    # 同上
    1103: "LIB",   # apply_global_style（纯 matplotlib）
    1111: "ENTRY", # inject_theme() 内部 st.markdown 注入 CSS → 必须每 rerun
    1314: "LIB",   # _preload_heavy_libraries() 幂等 sys 守卫，import 时启动更早
}

segments = []
for node in tree.body:
    start = node.lineno
    if getattr(node, "decorator_list", None):
        start = min(start, min(d.lineno for d in node.decorator_list))
    end = node.end_lineno
    cat = classify(node)
    if start in OVERRIDES:
        cat = OVERRIDES[start]
    segments.append({"start": start, "end": end, "cat": cat})

def seg_text(seg):
    return "\n".join(lines[seg["start"] - 1: seg["end"]])

# DROP 段：旧入口流程，由新导航区/尾部重建代码替代，不进入 lib 也不进入口
# （分类器无法识别这些纯函数调用是入口流程：page_home() 不含 st 引用）
def _is_drop(seg):
    t = seg_text(seg).strip()
    if t.startswith('if page == "🏠 首页":'):
        return True   # 旧 dispatch 链 → pg.run()
    if t == "page = render_sidebar()":
        return True   # 旧侧边栏 → 新导航区
    if t.startswith('_prev_page = st.session_state.get("_prev_page", None)'):
        return True   # 尾部重建
    if t.startswith("if _prev_page is not None and _prev_page != page:"):
        return True   # 尾部重建（置顶动画块）
    if t == 'st.session_state["_prev_page"] = page':
        return True   # 尾部重建
    if t == "render_top_status_bar()":
        return True   # 尾部重建
    if t.startswith("if _render_global_task_lock(page):"):
        return True   # 尾部重建
    if t == "_maybe_autosave_session()":
        return True   # 尾部重建
    return False

dropped = [s for s in segments if _is_drop(s)]
for dseg in dropped:
    dseg["cat"] = "DROP"
print(f"DROP 段: {len(dropped)} 个（旧 dispatch/侧边栏/尾部流程）")

print("分类统计:", Counter(s["cat"] for s in segments))
assert not any(s["cat"] == "REVIEW" for s in segments), "存在未裁决的 REVIEW 段！"
# ---------------------------------------------------------------- 备份
if not os.path.exists(BAK):
    shutil.copy2(SRC, BAK)
    print(f"已备份原文件 → {BAK}")

# ---------------------------------------------------------------- 1. app_lib.py
lib_parts = [
    '# -*- coding: utf-8 -*-',
    '"""',
    'CFRP 平台共享库（由 app.py 拆分自动生成）。',
    '',
    '包含原 app.py 的全部定义（函数/类/常量/幂等环境设置），',
    '每进程仅在首次 import 时执行一次；每 rerun 不再重复执行。',
    'UI 初始化与导航见 app.py 入口；页面封装见 app_pages/。',
    '"""',
    '',
]
for seg in segments:
    if seg["cat"] == "LIB":
        lib_parts.append(seg_text(seg))
        lib_parts.append("")

lib_src = "\n".join(lib_parts)

# page_home 旧导航机制 → st.switch_page（4 处）
_nav_repl = [
    ('st.session_state["_nav_to"] = "数据上传"\n            st.rerun()',
     'st.switch_page("app_pages/data_upload.py")'),
    ('st.session_state["_nav_to"] = "分子特征"\n            st.rerun()',
     'st.switch_page("app_pages/molecular_features.py")'),
    ('st.session_state["_nav_to"] = "模型训练"\n            st.rerun()',
     'st.switch_page("app_pages/model_training.py")'),
    ('st.session_state["_nav_to"] = "预测应用"\n            st.rerun()',
     'st.switch_page("app_pages/prediction.py")'),
]
for old, new in _nav_repl:
    assert old in lib_src, f"导航替换锚点缺失: {old[:50]}"
    lib_src = lib_src.replace(old, new)

# __all__（含下划线名，使 from app_lib import * 完整迁移命名空间）
lib_tree = ast.parse(lib_src)
names = set()
for node in lib_tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names.add(node.name)
    elif isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name):
                names.add(t.id)
            elif isinstance(t, (ast.Tuple, ast.List)):
                names.update(e.id for e in t.elts if isinstance(e, ast.Name))
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        names.add(node.target.id)
    elif isinstance(node, ast.Import):
        names.update(a.asname or a.name.split(".")[0] for a in node.names)
    elif isinstance(node, ast.ImportFrom):
        names.update(a.asname or a.name for a in node.names)
names.discard("*")
all_line = "__all__ = " + json.dumps(sorted(names), ensure_ascii=False, indent=0).replace("\n", " ")
lib_src += "\n\n# 转移期命名空间完整导出（含下划线名），页面与测试从本库取全部符号。\n" + all_line + "\n"

with open(LIB, "w", encoding="utf-8") as f:
    f.write(lib_src)
print(f"✅ {LIB}: {len(lib_src.splitlines())} 行")

# ---------------------------------------------------------------- 2. 安全检查：lib 顶层无 st 命令
bad_st = []
for node in ast.parse(lib_src).body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        continue
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        continue
    if uses_st(node):
        bad_st.append((node.lineno, "\n".join(lib_src.split("\n")[node.lineno-1:node.end_lineno])[:120]))
if bad_st:
    print("❌ app_lib 顶层存在 st 命令：")
    for ln, t in bad_st:
        print(f"  L{ln}: {t}")
    sys.exit(1)
print("✅ app_lib 顶层无 st 命令调用")

# ---------------------------------------------------------------- 3. 新入口 app.py
PAGE_DEFS = [
    ("数据准备", [
        ("home", "page_home", "🏠 首页"),
        ("data_upload", "page_data_upload", "📤 数据上传"),
        ("data_explore", "page_data_explore", "🔍 数据探索"),
        ("data_cleaning", "page_data_cleaning", "🧹 数据清洗"),
        ("data_enhancement", "page_data_enhancement", "✨ 数据增强"),
    ]),
    ("特征工程", [
        ("molecular_features", "page_molecular_features", "🧬 分子特征"),
        ("molecular_feature_reproduction", "page_molecular_feature_reproduction", "🧬 分子特征复现"),
        ("feature_registry", "page_feature_registry", "🧩 特征管理"),
        ("smiles_structure_tools", "page_smiles_structure_tools", "🧪 SMILES / BigSMILES 结构图像工具"),
        ("feature_selection", "page_feature_selection", "🎯 特征选择"),
    ]),
    ("建模分析", [
        ("model_training", "page_model_training", "🤖 模型训练"),
        ("training_records", "page_training_records", "📈 训练记录"),
        ("model_interpretation", "page_model_interpretation", "📊 模型解释"),
        ("hyperparameter_optimization", "page_hyperparameter_optimization", "⚙️ 超参优化"),
        ("active_learning", "page_active_learning", "🧠 主动学习"),
    ]),
    ("应用预测", [
        ("prediction", "page_prediction", "🔮 预测应用"),
        ("model_imputation", "page_model_imputation", "🔧 模型补齐数据"),
        ("virtual_screening", "page_virtual_screening", "🧪 虚拟分子筛选"),
    ]),
    ("系统记录", [
        ("status_log", "page_status_log", "📋 状态条记录"),
    ]),
]

nav_lines = []
nav_lines.append("_NAV_GROUPS = {")
for group, items in PAGE_DEFS:
    nav_lines.append(f'    "{group}": [')
    for url_path, _fn, title in items:
        nav_lines.append(
            f'        st.Page("app_pages/{url_path}.py", title="{title}", url_path="{url_path}"),'
        )
    nav_lines.append("    ],")
nav_lines.append("}")
nav_lines.append("pg = st.navigation(_NAV_GROUPS)")

# render_sidebar 函数体手术（文本锚点，行号稳健）
rs_start = None
for seg in segments:
    t = seg_text(seg)
    if t.startswith("def render_sidebar():"):
        rs_start = seg["start"]
        rs_end = seg["end"]
        break
assert rs_start, "未找到 render_sidebar"
rs_lines = lines[rs_start - 1: rs_end]

# 删除 def 行 + docstring
assert rs_lines[0].startswith("def render_sidebar():")
body = rs_lines[1:]
if body[0].strip().startswith('"""渲染侧边栏导航"""'):
    body = body[1:]

# 删除旧导航段：从 current_page = resolve_navigation_page( 到 _last_active_page 行
def find_idx(pred, seq):
    for i, l in enumerate(seq):
        if pred(l):
            return i
    return -1

i_start = find_idx(lambda l: l.strip().startswith("current_page = resolve_navigation_page("), body)
i_end = find_idx(lambda l: l.strip().startswith('st.session_state["_last_active_page"] = page'), body)
assert i_start > 0 and i_end > i_start, f"导航段锚点定位失败 {i_start} {i_end}"
# 三段式结构：
#   part1 = with st.sidebar: 块开头（title/caption/legacy cleanup）
#   middle = 模块层级 _NAV_GROUPS + pg = st.navigation（导航菜单自动挂侧边栏）
#   part2 = 第二个 with st.sidebar: 块（panels/数据状态/系统工具/...）
part1 = body[:i_start]
part1[0] = body[0].lstrip()  # “    with st.sidebar:” → 顶层 with（内容 8 空格缩进在块内合法）
part2 = body[i_end + 1:]

# part2 内删除 return page，并补回 active_task_lock（原在导航段内定义）
i_ret = find_idx(lambda l: l.strip() == "return page", part2)
assert i_ret > 0, "未找到 return page"
del part2[i_ret]
part2_wrapped = [
    "with st.sidebar:",
    "        active_task_lock = bool(get_task_manager().get_active_tasks())",
]
part2_wrapped.extend(part2)

# legacy cleanup 追加新 key（在 part1 中）
old_tuple_tail = '            "_last_app_page",\n        ):'
new_tuple_tail = ('            "_last_app_page",\n'
                  '            "nav_page",\n'
                  '            "_last_active_page",\n'
                  '            "_nav_to",\n        ):')
sidebar_text = (
    "".join(l + "\n" for l in part1)
    + "\n"
    + "\n".join(nav_lines)          # 模块层级，0 缩进
    + "\n\n"
    + "".join(l + "\n" for l in part2_wrapped)
)
assert old_tuple_tail in sidebar_text, "legacy 清理元组锚点缺失"
sidebar_text = sidebar_text.replace(old_tuple_tail, new_tuple_tail)

# ---------------------------------------------------------------- 组装入口
entry_parts = [
    '# -*- coding: utf-8 -*-',
    '"""',
    '碳纤维复合材料智能预测平台 —— 应用入口（由原 28000 行 app.py 拆分自动生成）。',
    '',
    '架构：',
    '  app_lib.py    共享库：全部定义与幂等环境设置（每进程一次）',
    '  app_pages/    19 个 st.Page 页面封装',
    '  本文件        每 rerun 执行：UI 初始化 + 分组导航 + 任务锁 + 自动保存',
    '"""',
    '',
    'from app_lib import *',
    '',
]
# ENTRY 段按原顺序
for seg in segments:
    if seg["cat"] == "ENTRY":
        entry_parts.append(seg_text(seg))
        entry_parts.append("")

entry_parts.append("# ============================================================")
entry_parts.append("# 页面导航（st.navigation 分组侧边栏）")
entry_parts.append("# ============================================================")
entry_parts.extend(["", "_NAV_GROUPS_UNUSED = None"])  # 占位，实际插入在侧边栏手术文本中

# 尾部（原样 + pg.run() 替代 dispatch）
tail_parts = []

# _prev_page 段（28115-28127 附近，用锚点从 segments ENTRY 中定位：assign referencing st 的 _prev_page）
# _prev_page 三段（均已被标记 DROP，从 dropped 中取原文本）：赋值读取 / 置顶动画 if 块 / 写入
prev_assign = next(
    (s for s in dropped if seg_text(s).strip().startswith('_prev_page = st.session_state.get')),
    None,
)
assert prev_assign is not None, "未找到 _prev_page 赋值段"
html_block = next(
    (s for s in dropped if seg_text(s).strip().startswith("if _prev_page is not None and _prev_page != page:")),
    None,
)
assert html_block is not None, "未找到置顶动画块"
prev_write = next(
    (s for s in dropped if seg_text(s).strip() == 'st.session_state["_prev_page"] = page'),
    None,
)
assert prev_write is not None, "未找到 _prev_page 写入段"
tail_parts.append("page = pg.title  # 与原 task-lock / 置顶逻辑兼容（title 含 emoji）")
tail_parts.append("\n".join(lines[prev_assign["start"] - 1: prev_assign["end"]]))
tail_parts.append("\n".join(lines[html_block["start"] - 1: html_block["end"]]))
tail_parts.append("\n".join(lines[prev_write["start"] - 1: prev_write["end"]]))
tail_parts.append("render_top_status_bar()")
tail_parts.append(
    "if _render_global_task_lock(page):\n"
    "    _maybe_autosave_session()\n"
    "    st.stop()"
)
tail_parts.append("pg.run()")
tail_parts.append("")
tail_parts.append("# 自动保存快照（断连保护）")
tail_parts.append("_maybe_autosave_session()")

entry_src = "\n".join(entry_parts)
# 把侧边栏手术文本插入（替换占位行；占位符可能是 join 的末项无尾随换行）
sidebar_block = "\n# ---------- 侧边栏（导航菜单在原 selectbox 位置） ----------\n" + sidebar_text + "\n"
if "\n_NAV_GROUPS_UNUSED = None" in entry_src:
    entry_src = entry_src.replace("\n_NAV_GROUPS_UNUSED = None", sidebar_block)
elif entry_src.endswith("_NAV_GROUPS_UNUSED = None"):
    entry_src = entry_src[: -len("_NAV_GROUPS_UNUSED = None")] + sidebar_block.lstrip("\n")
assert "_NAV_GROUPS_UNUSED" not in entry_src, "占位符未替换！"
assert "with st.sidebar:" in entry_src, "侧边栏文本未插入！"
entry_src += "\n# ============================================================\n# 主流程\n# ============================================================\n" + "\n".join(tail_parts) + "\n"

# 校验：入口不应包含旧 dispatch 链
assert 'elif page == "🔮 预测应用":' not in entry_src, "旧 dispatch 残留！"
assert "st.selectbox(\n            \"页面导航\"" not in entry_src, "旧 selectbox 残留！"

with open(SRC, "w", encoding="utf-8") as f:
    f.write(entry_src)
print(f"✅ 新入口 {SRC}: {len(entry_src.splitlines())} 行")

# ---------------------------------------------------------------- 4. app_pages/*.py
os.makedirs(PAGES_DIR, exist_ok=True)
init_file = os.path.join(PAGES_DIR, "__init__.py")
with open(init_file, "w", encoding="utf-8") as f:
    f.write("# -*- coding: utf-8 -*-\n# CFRP 平台页面封装包（st.Page 入口文件）\n")

count = 0
for group, items in PAGE_DEFS:
    for url_path, fn, title in items:
        page_src = (
            "# -*- coding: utf-8 -*-\n"
            f'"""页面：{title}（st.Page 薄封装，由拆分工具自动生成）"""\n\n'
            "from app_lib import *\n\n"
            f"{fn}()\n"
        )
        with open(os.path.join(PAGES_DIR, f"{url_path}.py"), "w", encoding="utf-8") as f:
            f.write(page_src)
        count += 1
print(f"✅ {PAGES_DIR}/: 生成 {count} 个页面文件")

# ---------------------------------------------------------------- 5. tests 导入迁移
import re
for tf in ("tests/test_app_scope_regressions.py", "tests/test_data_explore_export.py"):
    if not os.path.exists(tf):
        continue
    t = open(tf, encoding="utf-8").read()
    n = t.count("import app")
    t2 = t.replace("import app\n", "import app_lib as app\n")
    with open(tf, "w", encoding="utf-8") as f:
        f.write(t2)
    print(f"✅ {tf}: {n} 处 import app → import app_lib as app")

print("\n=== 生成完成 ===")
