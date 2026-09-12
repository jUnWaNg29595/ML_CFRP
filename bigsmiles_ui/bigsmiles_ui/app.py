from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from .batch_io import process_table, read_uploaded_table, write_table
    from .renderer import RenderOptions, RenderResult, added_columns, render_structure
except ImportError:  # Streamlit 直接执行 app.py 时没有包上下文
    import sys

    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from bigsmiles_ui.batch_io import process_table, read_uploaded_table, write_table
    from bigsmiles_ui.renderer import RenderOptions, RenderResult, added_columns, render_structure


def format_result_summary(result: RenderResult) -> str:
    return (
        f"解析状态：{result.parse_status}；类型：{result.detected_type}；"
        f"绘图状态：{result.draw_status}；渲染器：{result.renderer or '未解析'}"
    )


def detect_uploaded_table_columns(uploaded_file) -> tuple["pd.DataFrame", list[str]]:
    if pd is None:
        raise RuntimeError("当前 Python 环境未安装 pandas，请先安装 bigsmiles_ui/requirements.txt")
    frame = read_uploaded_table(uploaded_file.name, uploaded_file.getvalue())
    return frame, [str(column) for column in frame.columns]


def _default_output_name(file_name: str, suffix: str) -> str:
    stem = Path(file_name).stem or "结构检查结果"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stem}_结构检查_{timestamp}{suffix}"


def _result_text(result: RenderResult) -> str:
    lines = [
        f"原始结构：{result.raw_string}",
        f"识别类型：{result.detected_type}",
        f"解析状态：{result.parse_status}",
        f"规范化结构：{result.normalized_structure}",
        f"绘图状态：{result.draw_status}",
        f"错误信息：{result.error_message}",
        f"警告信息：{result.warning_message}",
        f"渲染器：{result.renderer}",
    ]
    return "\n".join(lines)


def _show_single_tab(st) -> None:
    st.subheader("单条结构渲染")
    raw = st.text_area(
        "输入 SMILES 或 BigSMILES",
        height=150,
        placeholder="例如：CCO，或 CC{[>][<]CC(C)[>][<]}CC(C)=C",
    )
    requested_type = st.selectbox("结构类型", ["自动识别", "SMILES", "BigSMILES"], key="single_type")
    type_map = {"自动识别": "auto", "SMILES": "smiles", "BigSMILES": "bigsmiles"}
    col1, col2 = st.columns(2)
    with col1:
        image_width = st.number_input("图片宽度", min_value=200, max_value=3000, value=1000, step=50)
    with col2:
        image_height = st.number_input("图片高度", min_value=200, max_value=3000, value=700, step=50)
    sample = st.checkbox("生成代表性采样链段示意图（RDKit 片段整体复制，不代表真实链长）", value=False)
    repeat_units = 5
    random_seed = 42
    if sample:
        repeat_units = st.number_input("重复单元数", min_value=1, max_value=50, value=5, step=1)
        random_seed = st.number_input("随机种子", min_value=0, max_value=2_147_483_647, value=42, step=1)
    if st.button("生成结构图", type="primary"):
        output_dir = Path(__file__).resolve().parent / "generated" / "single"
        options = RenderOptions(
            requested_type=type_map[requested_type],
            render_sample_chain=sample,
            repeat_units=int(repeat_units),
            random_seed=int(random_seed),
            image_width=int(image_width),
            image_height=int(image_height),
        )
        result = render_structure(raw, output_dir, options)
        if result.parse_status == "valid" and result.draw_status == "rendered":
            st.success(format_result_summary(result))
        elif result.parse_status == "valid":
            st.warning(format_result_summary(result))
        elif result.parse_status == "empty":
            st.info(format_result_summary(result))
        else:
            st.error(format_result_summary(result))
        if result.error_message:
            st.error(result.error_message)
        if result.warning_message:
            st.warning(result.warning_message)
        if result.normalized_structure:
            st.code(result.normalized_structure, language="text")
        if result.main_image_path:
            image_path = output_dir / result.main_image_path
            if image_path.exists():
                st.image(str(image_path), caption="主图：重复单元/连接模式视图或普通分子图")
                st.download_button(
                    "下载主图",
                    data=image_path.read_bytes(),
                    file_name=image_path.name,
                    mime="image/png",
                    key="download_main_image",
                )
        if result.sample_image_path:
            image_path = output_dir / result.sample_image_path
            if image_path.exists():
                st.image(str(image_path), caption="代表性采样链段图")
                st.download_button(
                    "下载采样图",
                    data=image_path.read_bytes(),
                    file_name=image_path.name,
                    mime="image/png",
                    key="download_sample_image",
                )
        st.download_button(
            "下载文本结果",
            data=_result_text(result).encode("utf-8"),
            file_name="结构检查结果.txt",
            mime="text/plain",
            key="download_result_text",
        )


def _show_batch_tab(st) -> None:
    st.subheader("批量结构检查")
    uploaded = st.file_uploader("上传 CSV 或 XLSX 文件", type=["csv", "xlsx", "xlsm"])
    if uploaded is None:
        st.info("请上传包含结构列的表格")
        return
    try:
        frame, columns = detect_uploaded_table_columns(uploaded)
    except Exception as exc:
        st.error(f"读取表格失败：{exc}")
        return
    if not columns:
        st.error("表格没有可用列")
        return
    structure_column = st.selectbox("选择结构列", columns, key="batch_structure_column")
    requested_type_label = st.selectbox("结构类型", ["自动识别", "SMILES", "BigSMILES"], key="batch_type")
    type_map = {"自动识别": "auto", "SMILES": "smiles", "BigSMILES": "bigsmiles"}
    sample = st.checkbox("生成代表性采样链段示意图（RDKit 片段整体复制，不代表真实链长）", value=False, key="batch_sample")
    repeat_units = 5
    random_seed = 42
    if sample:
        repeat_units = st.number_input("批量重复单元数", min_value=1, max_value=50, value=5, step=1, key="batch_repeat")
        random_seed = st.number_input("批量随机种子", min_value=0, max_value=2_147_483_647, value=42, step=1, key="batch_seed")
    output_format = st.selectbox("输出格式", ["CSV", "XLSX"], key="batch_format")
    default_suffix = ".csv" if output_format == "CSV" else ".xlsx"
    output_name = st.text_input("结果文件名", value=_default_output_name(uploaded.name, default_suffix))
    if not output_name.lower().endswith((".csv", ".xlsx")):
        output_name += default_suffix
    if not st.button("开始批量检查", type="primary"):
        st.caption(f"已读取 {len(frame)} 行；原始列不会被覆盖。")
        return
    root = Path(__file__).resolve().parent / "generated" / datetime.now().strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    options = RenderOptions(
        requested_type=type_map[requested_type_label],
        render_sample_chain=sample,
        repeat_units=int(repeat_units),
        random_seed=int(random_seed),
    )
    progress = st.progress(0, text="准备开始")
    status = st.empty()
    counts = {"valid": 0, "invalid": 0, "empty": 0, "valid_but_not_renderable": 0}

    def update(current: int, total: int, result: RenderResult) -> None:
        counts[result.parse_status] = counts.get(result.parse_status, 0) + 1
        if result.draw_status == "valid_but_not_renderable":
            counts["valid_but_not_renderable"] += 1
        progress.progress(current / max(total, 1), text=f"正在处理：第 {current} / {total} 行")
        status.info(
            f"进度 {current}/{total}；有效 {counts['valid']}；无效 {counts['invalid']}；"
            f"空值 {counts['empty']}；可解析但不可绘图 {counts['valid_but_not_renderable']}"
        )

    try:
        result_frame = process_table(frame, structure_column, root, options, update)
        output_path = write_table(result_frame, root / output_name)
    except Exception as exc:
        progress.empty()
        status.empty()
        st.error(f"批量处理失败：{exc}")
        return
    progress.progress(1.0, text="批量检查完成")
    status.success(
        f"完成：共 {len(result_frame)} 行；有效 {counts['valid']}；无效 {counts['invalid']}；空值 {counts['empty']}。"
    )
    st.dataframe(result_frame.head(100), use_container_width=True)
    st.download_button(
        "下载结果表",
        data=output_path.read_bytes(),
        file_name=output_path.name,
        mime="text/csv" if output_path.suffix.lower() == ".csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_batch_table",
    )
    st.caption(f"图片位于结果目录：{root / 'images'}；移动结果表时请保留 images 文件夹。")


def main() -> None:
    try:
        import streamlit as st
    except Exception as exc:
        raise SystemExit(f"无法启动页面：缺少 Streamlit 依赖，请安装 bigsmiles_ui/requirements.txt。{exc}") from exc
    st.set_page_config(page_title="BigSMILES / SMILES 结构可视化", layout="wide")
    st.title("BigSMILES / SMILES 结构可视化")
    st.caption("本工具只在本机解析和绘图，不调用文献提取 API，不修改现有提取结果。")
    single_tab, batch_tab = st.tabs(["单条渲染", "批量检查"])
    with single_tab:
        _show_single_tab(st)
    with batch_tab:
        _show_batch_tab(st)


if __name__ == "__main__":
    main()
