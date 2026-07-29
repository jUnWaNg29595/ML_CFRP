from __future__ import annotations


NAVIGATION_GROUPS = (
    (
        "数据准备",
        (
            "🏠 首页",
            "📤 数据上传",
            "🔍 数据探索",
            "🧹 数据清洗",
            "✨ 数据增强",
        ),
    ),
    (
        "特征工程",
        (
            "🧬 分子特征",
            "🧬 分子特征复现",
            "🖼️ 图像转SMILES",
            "🎯 特征选择",
        ),
    ),
    (
        "建模分析",
        (
            "🤖 模型训练",
            "📈 训练记录",
            "📊 模型解释",
            "⚙️ 超参优化",
            "🧠 主动学习",
        ),
    ),
    (
        "应用预测",
        (
            "🔮 预测应用",
            "🔧 模型补齐数据",
            "🧪 虚拟分子筛选",
        ),
    ),
    (
        "系统记录",
        (
            "📋 状态条记录",
        ),
    ),
)

NAVIGATION_PAGES = tuple(
    page
    for _group_name, group_pages in NAVIGATION_GROUPS
    for page in group_pages
)

NAVIGATION_ALIASES = {
    page.split(" ", 1)[1]: page
    for page in NAVIGATION_PAGES
    if " " in page
}
NAVIGATION_ALIASES.update(
    {
        "首页": "🏠 首页",
        "分子特征复现": "🧬 分子特征复现",
        "图像转SMILES": "🖼️ 图像转SMILES",
        "超参优化": "⚙️ 超参优化",
        "主动学习": "🧠 主动学习",
        "预测应用": "🔮 预测应用",
        "模型补齐数据": "🔧 模型补齐数据",
        "虚拟分子筛选": "🧪 虚拟分子筛选",
    }
)


def resolve_navigation_page(
    current_page: str | None = None,
    pending_page: str | None = None,
) -> str:
    candidate = pending_page if pending_page else current_page
    if candidate in NAVIGATION_PAGES:
        return candidate
    if candidate in NAVIGATION_ALIASES:
        return NAVIGATION_ALIASES[candidate]
    if current_page in NAVIGATION_PAGES:
        return current_page
    if current_page in NAVIGATION_ALIASES:
        return NAVIGATION_ALIASES[current_page]
    return "🏠 首页"
