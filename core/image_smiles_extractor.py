# -*- coding: utf-8 -*-
"""
Image/File -> SMILES extractor wrapper for the CFRP ML platform.

- Primary engine: DECIMER (Image Transformer) for chemical structure depictions.
- Supports image files (png/jpg/jpeg/bmp/tif/tiff/webp/heif/heic) and (optionally) PDFs.
- Designed to be OPTIONAL: if TensorFlow/DECIMER deps are missing, the app will show install hints.

Usage:
    from core.image_smiles_extractor import decimer_is_available, smiles_from_path
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

import zipfile
from functools import lru_cache

# ------------------------------------------------------------
# DECIMER model cache repair helpers
# ------------------------------------------------------------
def _decimer_model_zip_candidates() -> list[str]:
    """Common locations for DECIMER-V2 cached model zip."""
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, ".data", "DECIMER-V2", "models.zip"),   # Linux / WSL (observed)
        os.path.join(home, "DECIMER-V2", "models.zip"),           # fallback
        os.path.join(home, ".cache", "DECIMER-V2", "models.zip"), # some setups
    ]
    # Windows-style fallbacks (if running native Windows python)
    candidates.append(os.path.join(home, ".data", "DECIMER-V2", "models.zip"))
    candidates.append(os.path.join(home, "AppData", "Local", "DECIMER-V2", "models.zip"))
    return candidates


def _repair_corrupt_decimer_zip() -> None:
    """Delete a corrupted models.zip so DECIMER can re-download cleanly."""
    for p in _decimer_model_zip_candidates():
        try:
            if os.path.exists(p) and (not zipfile.is_zipfile(p)):
                # Corrupted / partial download
                os.remove(p)
        except Exception:
            pass


@lru_cache(maxsize=1)
def _get_decimer_module():
    """Import DECIMER once per process (Streamlit reruns won't re-download models)."""
    _ensure_decimer_on_path()

    # If previous downloads were interrupted, models.zip might be corrupt.
    _repair_corrupt_decimer_zip()

    # Import tensorflow first (DECIMER relies on it)
    import tensorflow as tf  # noqa: F401

    # Importing DECIMER may trigger (first-time) model download + unzip.
    import DECIMER.decimer as decimer
    return decimer

# ------------------------------------------------------------
# Optional imports
# ------------------------------------------------------------
_DECIMER_IMPORT_ERROR: Optional[Exception] = None


def _ensure_decimer_on_path():
    """
    DECIMER is vendored into the project root as a 'DECIMER' package.
    In most cases Streamlit runs with project root in sys.path already.
    This helper exists for edge cases (e.g., running from another cwd).
    """
    try:
        import DECIMER  # noqa: F401
        return
    except Exception:
        pass

    # Add project root (one level up from this file) to sys.path
    try:
        import sys
        here = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(here, os.pardir))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
    except Exception:
        # Best-effort only
        return


def decimer_is_available() -> Tuple[bool, str]:
    """
    Check whether DECIMER + TensorFlow deps are available.

    Returns (available, message).
    """
    global _DECIMER_IMPORT_ERROR
    try:
        _get_decimer_module()
        return True, "DECIMER is available."
    except Exception as e:
        # Auto-repair: if cached zip is corrupted, remove and retry once
        msg = str(e)
        if "zip" in msg.lower() and "not a zip" in msg.lower():
            try:
                _repair_corrupt_decimer_zip()
                _get_decimer_module.cache_clear()
                _get_decimer_module()
                return True, "DECIMER is available."
            except Exception as e2:
                _DECIMER_IMPORT_ERROR = e2
                return False, f"DECIMER not available: {e2}"
        _DECIMER_IMPORT_ERROR = e
        return False, f"DECIMER not available: {e}"



@dataclass
class SmilesPrediction:
    filename: str
    smiles: str
    confidence: Optional[float] = None
    engine: str = "DECIMER"
    page_index: Optional[int] = None


def _is_number(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def normalize_decimer_output(pred, want_confidence: bool = False) -> tuple[str, Optional[float]]:
    """Normalize DECIMER predict_SMILES output to (smiles, confidence).

    DECIMER may return:
    - str: SMILES
    - list[(smiles, conf)]: top-k
    - list[(token, conf)]: token-level (char-level) output when confidence=True in some versions
    - tuple(smiles, token_confs) or other variants
    This function makes the UI output stable and readable.
    """
    # 1) simple string
    if isinstance(pred, str):
        return pred, None
    if isinstance(pred, bytes):
        try:
            return pred.decode("utf-8", errors="ignore"), None
        except Exception:
            return str(pred), None

    # 2) tuple variants
    if isinstance(pred, tuple) and len(pred) >= 1:
        if isinstance(pred[0], str):
            smiles = pred[0]
            conf = None
            if want_confidence and len(pred) >= 2:
                # (smiles, conf)
                if _is_number(pred[1]):
                    conf = float(pred[1])
                # (smiles, [(token, conf), ...])
                elif isinstance(pred[1], list) and pred[1] and all(isinstance(t, tuple) and len(t) == 2 and _is_number(t[1]) for t in pred[1]):
                    try:
                        conf = float(np.mean([float(t[1]) for t in pred[1]]))
                    except Exception:
                        conf = None
            return smiles, conf

    # 3) list variants
    if isinstance(pred, list) and len(pred) > 0:
        p0 = pred[0]

        # 3.1) top-k: [(smiles, conf), ...]
        if isinstance(p0, tuple) and len(p0) == 2 and isinstance(p0[0], str) and _is_number(p0[1]) and len(p0[0]) > 1:
            return p0[0], float(p0[1]) if want_confidence else None

        # 3.2) token-level: [(token, conf), ...] where token may be 1-char or subword pieces
        if all(isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str) and _is_number(x[1]) for x in pred):
            smiles = "".join([x[0] for x in pred])
            conf = None
            if want_confidence:
                try:
                    conf = float(np.mean([float(x[1]) for x in pred]))
                except Exception:
                    conf = None
            return smiles, conf

        # 3.3) list[str]
        if isinstance(p0, str):
            return p0, None

    # 4) fallback
    return str(pred), None


def _read_pdf_to_images(pdf_path: str) -> List[np.ndarray]:
    """
    Convert PDF pages to RGB numpy arrays.

    Preferred backend: PyMuPDF (fitz).
    Fallback: pdf2image (requires poppler installed).

    Returns: list of HxWx3 uint8 arrays.
    """
    # --- PyMuPDF ---
    try:
        import fitz  # type: ignore
        doc = fitz.open(pdf_path)
        images: List[np.ndarray] = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap(alpha=False)  # RGB
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            images.append(img)
        doc.close()
        return images
    except Exception:
        pass

    # --- pdf2image ---
    try:
        from pdf2image import convert_from_path  # type: ignore
        pil_images = convert_from_path(pdf_path, dpi=300)
        out: List[np.ndarray] = []
        for im in pil_images:
            im = im.convert("RGB")
            out.append(np.array(im))
        return out
    except Exception as e:
        raise RuntimeError(
            "PDF support requires PyMuPDF (fitz) or pdf2image + poppler. "
            "Install one of them, e.g. `pip install pymupdf`."
        ) from e


def smiles_from_image_array(
    image: np.ndarray,
    confidence: bool = False,
    hand_drawn: bool = False,
) -> Union[str, List[Tuple[str, float]]]:
    """
    Predict SMILES from an image numpy array (HxWxC).
    When confidence=True, returns list[(smiles, confidence)] (top-k from DECIMER).
    """
    ok, msg = decimer_is_available()
    if not ok:
        raise RuntimeError(msg)

    decimer = _get_decimer_module()

    pred = decimer.predict_SMILES(image, confidence=confidence, hand_drawn=hand_drawn)
    smiles, conf = normalize_decimer_output(pred, want_confidence=bool(confidence))
    return ([(smiles, conf)] if confidence else smiles)


def smiles_from_path(
    file_path: str,
    confidence: bool = False,
    hand_drawn: bool = False,
) -> List[SmilesPrediction]:
    """
    Predict SMILES from a local file path (image or pdf).
    Returns a list because PDFs can yield multiple pages.
    """
    ext = os.path.splitext(file_path)[1].lower().strip(".")
    fname = os.path.basename(file_path)

    # PDF
    if ext == "pdf":
        images = _read_pdf_to_images(file_path)
        preds: List[SmilesPrediction] = []
        for i, img in enumerate(images):
            pred = smiles_from_image_array(img, confidence=confidence, hand_drawn=hand_drawn)
            if confidence and isinstance(pred, list) and len(pred) > 0:
                preds.append(SmilesPrediction(filename=fname, smiles=pred[0][0], confidence=float(pred[0][1]), page_index=i))
            else:
                preds.append(SmilesPrediction(filename=fname, smiles=str(pred), confidence=None, page_index=i))
        return preds

    # Image
    ok, msg = decimer_is_available()
    if not ok:
        raise RuntimeError(msg)

    # Use cached import (avoid repeated downloads on Streamlit reruns)
    decimer = _get_decimer_module()
    pred = decimer.predict_SMILES(file_path, confidence=confidence, hand_drawn=hand_drawn)

    smiles, conf = normalize_decimer_output(pred, want_confidence=bool(confidence))
    return [SmilesPrediction(filename=fname, smiles=smiles, confidence=conf if confidence else None)]


def smiles_from_bytes(
    file_bytes: bytes,
    filename: str,
    confidence: bool = False,
    hand_drawn: bool = False,
) -> List[SmilesPrediction]:
    """
    Predict SMILES from uploaded bytes (Streamlit uploader etc.)
    """
    suffix = os.path.splitext(filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        return smiles_from_path(tmp_path, confidence=confidence, hand_drawn=hand_drawn)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
