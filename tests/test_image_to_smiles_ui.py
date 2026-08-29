import base64
import io

import pytest
from PIL import Image


def _png_bytes(size=(2400, 1200)):
    image = Image.new("RGB", size, "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_clipboard_payload_decodes_image_and_preserves_filename(monkeypatch):
    import app

    raw = _png_bytes()
    payload = {
        "data_b64": base64.b64encode(raw).decode("ascii"),
        "mime": "image/png",
        "name": "clipboard-structure.png",
        "size": len(raw),
    }

    decoded = app._decode_clipboard_image_payload(payload, max_bytes=len(raw) + 1)

    assert decoded == {
        "name": "clipboard-structure.png",
        "mime": "image/png",
        "data": raw,
    }


def test_clipboard_payload_rejects_non_image_and_oversized_data():
    import app

    raw = _png_bytes((64, 64))
    encoded = base64.b64encode(raw).decode("ascii")

    with pytest.raises(ValueError, match="图片"):
        app._decode_clipboard_image_payload(
            {"data_b64": encoded, "mime": "text/plain", "name": "bad.txt"},
            max_bytes=10_000,
        )

    with pytest.raises(ValueError, match="过大"):
        app._decode_clipboard_image_payload(
            {"data_b64": encoded, "mime": "image/png", "name": "large.png"},
            max_bytes=len(raw) - 1,
        )


def test_image_preview_is_bounded_without_mutating_source():
    import app

    raw = _png_bytes()
    preview = app._build_image_preview(raw, max_width=640, max_height=480)

    assert preview != raw
    with Image.open(io.BytesIO(preview)) as image:
        assert image.width <= 640
        assert image.height <= 480
    with Image.open(io.BytesIO(raw)) as source:
        assert source.size == (2400, 1200)


def test_image_to_smiles_page_uses_clipboard_component_and_bounded_preview():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(
        encoding="utf-8-sig"
    )
    start = source.index("def page_image_to_smiles():")
    page_source = source[start:]

    assert "_image_clipboard_component" in page_source
    assert "_decode_clipboard_image_payload" in page_source
    assert "_build_image_preview" in page_source
    assert "max_width=640" in page_source


def test_clipboard_component_separates_paste_focus_from_file_picker():
    from pathlib import Path

    component_source = (
        Path(__file__).resolve().parents[1] / "components" / "image_clipboard" / "index.html"
    ).read_text(encoding="utf-8")

    assert 'id="choose-button"' in component_source
    assert 'zone.addEventListener("click", () => zone.focus());' in component_source
    assert 'chooseButton.addEventListener("click"' in component_source
