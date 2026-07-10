# -*- coding: utf-8 -*-
"""Plot export utilities

- Convert matplotlib figures to PNG bytes (for Streamlit download buttons)
- Convert matplotlib figures to self-contained HTML (embed PNG as base64)
- Export DataFrame to CSV bytes (utf-8-sig)

This module is intentionally lightweight and has no Streamlit dependency.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import pandas as pd


def fig_to_png_bytes(fig, dpi: int = 160) -> bytes:
    """Convert a matplotlib figure to PNG bytes."""
    if fig is None:
        return b""
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    except Exception:
        # fallback
        fig.savefig(buf, format="png", dpi=dpi)
    return buf.getvalue()


def fig_to_html(fig, title: str = "figure", dpi: int = 160) -> str:
    """Convert a matplotlib figure to a standalone HTML string (with embedded PNG)."""
    png = fig_to_png_bytes(fig, dpi=dpi)
    b64 = base64.b64encode(png).decode("utf-8") if png else ""
    safe_title = (title or "figure").replace("<", "").replace(">", "")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{safe_title}</title>"
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>"
        "</head><body style='margin:0;padding:12px;font-family:Arial, sans-serif;'>"
        f"<h3 style='margin:0 0 12px 0;'>{safe_title}</h3>"
        f"<img src='data:image/png;base64,{b64}' style='max-width:100%;height:auto;border:1px solid #ddd;border-radius:6px;'>"
        "</body></html>"
    )


def df_to_csv_bytes(df: Optional[pd.DataFrame]) -> bytes:
    """Export DataFrame to CSV bytes (utf-8-sig)."""
    if df is None:
        return b""
    try:
        return df.to_csv(index=False).encode("utf-8-sig")
    except Exception:
        return df.to_csv(index=False).encode("utf-8")
