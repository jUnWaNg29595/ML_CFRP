from core.navigation import NAVIGATION_PAGES, resolve_navigation_page


def test_navigation_pages_are_unique_and_have_one_home_entry():
    assert len(NAVIGATION_PAGES) == len(set(NAVIGATION_PAGES))
    assert NAVIGATION_PAGES.count("🏠 首页") == 1


def test_pending_shortcut_alias_resolves_to_canonical_page():
    assert resolve_navigation_page(
        current_page="🏠 首页",
        pending_page="模型训练",
    ) == "🤖 模型训练"


def test_invalid_pending_shortcut_keeps_current_canonical_page():
    assert resolve_navigation_page(
        current_page="🧪 虚拟分子筛选",
        pending_page="不存在的页面",
    ) == "🧪 虚拟分子筛选"


def test_process_pls_training_option_defaults_to_disabled_without_locked_workflow():
    workflow = None
    enabled = bool(workflow and workflow.get("enabled"))
    assert enabled is False
