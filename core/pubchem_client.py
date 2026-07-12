# -*- coding: utf-8 -*-
"""Minimal PubChem PUG-REST client for SMILES harvesting."""

from __future__ import annotations

import json
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from http.client import IncompleteRead
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

try:
    from rdkit import Chem

    RDKIT_AVAILABLE = True
except Exception:
    Chem = None
    RDKIT_AVAILABLE = False

import numpy as np
import pandas as pd

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "pubchem"


def _normalize_worker_count(requested: int, task_count: int, cap: int = 6) -> int:
    try:
        workers = int(requested)
    except Exception:
        workers = 1
    workers = max(1, workers)
    task_count = max(1, int(task_count))
    return max(1, min(workers, task_count, int(cap)))


def _request_json(
    url: str,
    data: Optional[Dict[str, str]] = None,
    timeout: int = 30,
    retries: int = 2,
    backoff: float = 1.0,
) -> Dict:
    payload = None
    if data:
        payload = urlencode(data).encode("utf-8")
    headers = {
        "User-Agent": "ML-CFRP-HTVS/1.0",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    last_err = None
    for attempt in range(max(1, int(retries) + 1)):
        try:
            req = Request(url, data=payload, headers=headers)
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            return json.loads(raw)
        except (HTTPError, URLError, json.JSONDecodeError, IncompleteRead, TimeoutError, OSError) as exc:
            detail = ""
            if isinstance(exc, HTTPError):
                try:
                    detail = exc.read().decode("utf-8", errors="ignore")
                except Exception:
                    detail = ""
            if detail:
                last_err = RuntimeError(f"HTTP Error {getattr(exc, 'code', '')}: {detail.strip()[:500]}")
            else:
                last_err = exc
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
                continue
            raise RuntimeError(f"PubChem request failed: {last_err}") from exc
    raise RuntimeError(f"PubChem request failed: {last_err}")


def _build_smiles_cache_path(smarts: str, max_cids: int) -> Path:
    payload = json.dumps(
        {
            "query": _clean_query_text(smarts),
            "max_cids": int(max_cids),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return PUBCHEM_CACHE_DIR / f"{digest}.csv"


def _load_smiles_cache(smarts: str, max_cids: int) -> Optional[pd.DataFrame]:
    path = _build_smiles_cache_path(smarts, max_cids)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return pd.DataFrame(columns=["cid", "smiles", "mol_wt"])
    for col in ("cid", "smiles", "mol_wt"):
        if col not in df.columns:
            return None
    df = df[["cid", "smiles", "mol_wt"]].copy()
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df = df[df["smiles"].astype(bool)].drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    return df


def _save_smiles_cache(smarts: str, max_cids: int, df: pd.DataFrame) -> None:
    try:
        PUBCHEM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_df = (df if isinstance(df, pd.DataFrame) else pd.DataFrame()).copy()
        if cache_df.empty:
            cache_df = pd.DataFrame(columns=["cid", "smiles", "mol_wt"])
        else:
            for col in ("cid", "smiles", "mol_wt"):
                if col not in cache_df.columns:
                    cache_df[col] = np.nan
            cache_df = cache_df[["cid", "smiles", "mol_wt"]].copy()
        cache_df.to_csv(_build_smiles_cache_path(smarts, max_cids), index=False, encoding="utf-8-sig")
    except Exception:
        return


def _clean_query_text(text: str) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""
    s = s.replace("\n", "").replace("\r", "").strip()
    if s.lower().startswith("smarts:"):
        s = s.split(":", 1)[-1].strip()
    if s.lower().startswith("smiles:"):
        s = s.split(":", 1)[-1].strip()
    s = "".join(s.split())
    return s


def _extract_cids(payload: Dict) -> Tuple[List[int], Optional[str]]:
    if not payload:
        return [], None
    if "IdentifierList" in payload and payload["IdentifierList"].get("CID"):
        return payload["IdentifierList"]["CID"], None
    if "Waiting" in payload and payload["Waiting"].get("ListKey"):
        return [], payload["Waiting"]["ListKey"]
    if "ListKey" in payload:
        return [], payload.get("ListKey")
    return [], None


def _is_plain_smiles_query(text: str) -> bool:
    if not text or not RDKIT_AVAILABLE:
        return False
    try:
        return Chem.MolFromSmiles(text) is not None
    except Exception:
        return False


def _filter_pubchem_query_matches(df: pd.DataFrame, query_text: str) -> pd.DataFrame:
    """Revalidate PubChem substructure results locally before caching or screening."""
    if not isinstance(df, pd.DataFrame) or df.empty or "smiles" not in df.columns:
        out = pd.DataFrame(columns=["cid", "smiles", "mol_wt"]) if df is None else df.copy()
        out.attrs.update({"query_validated": False, "raw_count": int(len(out)), "rejected_count": 0})
        return out
    if not RDKIT_AVAILABLE:
        out = df.copy()
        out.attrs.update({"query_validated": False, "raw_count": int(len(out)), "rejected_count": 0})
        return out

    query = _clean_query_text(query_text)
    try:
        query_mol = Chem.MolFromSmiles(query) if _is_plain_smiles_query(query) else Chem.MolFromSmarts(query)
    except Exception:
        query_mol = None
    if query_mol is None:
        out = df.copy()
        out.attrs.update({"query_validated": False, "raw_count": int(len(out)), "rejected_count": 0})
        return out

    keep_mask = []
    for value in df["smiles"].tolist():
        try:
            mol = Chem.MolFromSmiles(str(value or "").strip())
            keep_mask.append(bool(mol is not None and mol.HasSubstructMatch(query_mol)))
        except Exception:
            keep_mask.append(False)
    out = df.loc[keep_mask].drop_duplicates(subset=["smiles"]).reset_index(drop=True).copy()
    out.attrs.update(
        {
            "query_validated": True,
            "raw_count": int(len(df)),
            "rejected_count": int(len(df) - len(out)),
        }
    )
    return out


def _poll_listkey_for_cids(
    listkey: Optional[str],
    *,
    max_cids: int,
    timeout: int,
    retries: int,
    poll_delay: float,
    poll_timeout: float,
) -> List[int]:
    if not listkey:
        return []
    active_listkey = str(listkey).strip()
    if not active_listkey:
        return []
    start = time.time()
    while time.time() - start < float(poll_timeout):
        try:
            payload = _request_json(
                f"{PUBCHEM_BASE}/compound/listkey/{active_listkey}/cids/JSON",
                timeout=timeout,
                retries=retries,
            )
        except RuntimeError:
            break
        cids, next_listkey = _extract_cids(payload)
        if cids:
            return cids[: int(max_cids)]
        if next_listkey and str(next_listkey).strip():
            active_listkey = str(next_listkey).strip()
        time.sleep(float(poll_delay))
    return []


def fetch_cids_by_smarts(
    smarts: str,
    max_cids: int = 5000,
    timeout: int = 30,
    retries: int = 2,
    poll_delay: float = 1.0,
    poll_timeout: float = 60.0,
) -> List[int]:
    smarts = _clean_query_text(smarts)
    if not smarts:
        return []
    if _is_plain_smiles_query(smarts):
        return fetch_cids_by_smiles(
            smiles=smarts,
            max_cids=max_cids,
            timeout=timeout,
            retries=retries,
            poll_delay=poll_delay,
            poll_timeout=poll_timeout,
            search_mode="fastsubstructure",
        )
    encoded = quote(smarts, safe="")
    url = f"{PUBCHEM_BASE}/compound/substructure/smarts/{encoded}/cids/JSON"
    try:
        payload = _request_json(url, timeout=timeout, retries=retries)
    except RuntimeError as err:
        try:
            payload = _request_json(
                f"{PUBCHEM_BASE}/compound/substructure/smarts/JSON",
                data={"smarts": smarts},
                timeout=timeout,
                retries=retries,
            )
        except RuntimeError:
            if RDKIT_AVAILABLE:
                try:
                    mol = Chem.MolFromSmarts(smarts)
                    if mol is not None:
                        smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                        return fetch_cids_by_smiles(
                            smiles=smi,
                            max_cids=max_cids,
                            timeout=timeout,
                            retries=retries,
                            poll_delay=poll_delay,
                            poll_timeout=poll_timeout,
                        )
                except Exception:
                    pass
            raise err
    cids, listkey = _extract_cids(payload)
    if cids:
        return cids[: int(max_cids)]
    return _poll_listkey_for_cids(
        listkey,
        max_cids=max_cids,
        timeout=timeout,
        retries=retries,
        poll_delay=poll_delay,
        poll_timeout=poll_timeout,
    )


def fetch_cids_by_smiles(
    smiles: str,
    max_cids: int = 5000,
    timeout: int = 30,
    retries: int = 2,
    poll_delay: float = 1.0,
    poll_timeout: float = 60.0,
    search_mode: str = "exact",
) -> List[int]:
    smiles = _clean_query_text(smiles)
    if not smiles:
        return []
    encoded = quote(smiles, safe="")
    mode = str(search_mode or "exact").strip().lower()
    if mode == "fastsubstructure":
        url = f"{PUBCHEM_BASE}/compound/fastsubstructure/smiles/{encoded}/cids/JSON"
        post_url = None
        post_data = None
    elif mode == "substructure":
        url = f"{PUBCHEM_BASE}/compound/substructure/smiles/{encoded}/cids/JSON"
        post_url = f"{PUBCHEM_BASE}/compound/substructure/smiles/JSON"
        post_data = {"smiles": smiles}
    elif mode == "fastidentity":
        url = f"{PUBCHEM_BASE}/compound/fastidentity/smiles/{encoded}/cids/JSON"
        post_url = f"{PUBCHEM_BASE}/compound/fastidentity/smiles/cids/JSON"
        post_data = {"smiles": smiles}
    else:
        url = f"{PUBCHEM_BASE}/compound/smiles/{encoded}/cids/JSON"
        post_url = f"{PUBCHEM_BASE}/compound/smiles/cids/JSON"
        post_data = {"smiles": smiles}
    try:
        payload = _request_json(url, timeout=timeout, retries=retries)
    except RuntimeError:
        if not post_url:
            return []
        payload = _request_json(
            post_url,
            data=post_data,
            timeout=timeout,
            retries=retries,
        )
    cids, listkey = _extract_cids(payload)
    if cids:
        return cids[: int(max_cids)]
    return _poll_listkey_for_cids(
        listkey,
        max_cids=max_cids,
        timeout=timeout,
        retries=retries,
        poll_delay=poll_delay,
        poll_timeout=poll_timeout,
    )


def _chunked(items: Sequence[int], size: int) -> Iterable[List[int]]:
    step = max(1, int(size))
    for i in range(0, len(items), step):
        yield list(items[i : i + step])


def fetch_properties_by_cids(
    cids: Sequence[int],
    properties: Sequence[str],
    timeout: int = 30,
    retries: int = 2,
    max_workers: int = 4,
) -> pd.DataFrame:
    if not cids:
        return pd.DataFrame()
    props = ",".join([p for p in properties if p])
    chunks = list(_chunked(list(cids), 100))
    rows = []

    def _fetch_chunk(chunk: Sequence[int]):
        cid_str = ",".join(str(c) for c in chunk)
        url = f"{PUBCHEM_BASE}/compound/cid/{cid_str}/property/{props}/JSON"
        try:
            payload = _request_json(url, timeout=timeout, retries=retries)
            return payload.get("PropertyTable", {}).get("Properties", [])
        except RuntimeError:
            if len(chunk) <= 10:
                raise
            mid = max(1, len(chunk) // 2)
            left_rows = _fetch_chunk(chunk[:mid])
            right_rows = _fetch_chunk(chunk[mid:])
            return list(left_rows) + list(right_rows)

    worker_count = _normalize_worker_count(max_workers, len(chunks), cap=6)
    if worker_count <= 1 or len(chunks) <= 1:
        for chunk in chunks:
            table = _fetch_chunk(chunk)
            if table:
                rows.extend(table)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(_fetch_chunk, chunk): idx
                for idx, chunk in enumerate(chunks)
            }
            chunk_tables: List[Tuple[int, List[Dict]]] = []
            for future in as_completed(future_map):
                idx = future_map[future]
                table = future.result()
                if table:
                    chunk_tables.append((idx, table))
            for _, table in sorted(chunk_tables, key=lambda x: x[0]):
                rows.extend(table)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _fetch_smiles_by_smarts_uncached(
    smarts: str,
    max_cids: int = 5000,
    timeout: int = 30,
    retries: int = 2,
    property_workers: int = 4,
) -> pd.DataFrame:
    if _is_plain_smiles_query(smarts):
        cids = fetch_cids_by_smiles(
            smiles=smarts,
            max_cids=max_cids,
            timeout=timeout,
            retries=retries,
            search_mode="fastsubstructure",
        )
    else:
        cids = fetch_cids_by_smarts(
            smarts=smarts,
            max_cids=max_cids,
            timeout=timeout,
            retries=retries,
        )
    if not cids:
        return pd.DataFrame(columns=["cid", "smiles", "mol_wt"])
    df = fetch_properties_by_cids(
        cids,
        properties=["CanonicalSMILES", "ConnectivitySMILES", "IsomericSMILES", "MolecularWeight"],
        timeout=timeout,
        retries=retries,
        max_workers=property_workers,
    )
    if df.empty:
        return pd.DataFrame(columns=["cid", "smiles", "mol_wt"])

    out = pd.DataFrame()
    if "CID" in df.columns:
        out["cid"] = df["CID"]
    elif "cid" in df.columns:
        out["cid"] = df["cid"]
    else:
        out["cid"] = np.arange(1, len(df) + 1)

    smiles_sources = [
        "CanonicalSMILES",
        "ConnectivitySMILES",
        "IsomericSMILES",
        "SMILES",
        "smiles",
    ]
    smiles_series = None
    for col in smiles_sources:
        if col in df.columns:
            cand = df[col]
            if smiles_series is None:
                smiles_series = cand
            else:
                smiles_series = smiles_series.fillna(cand)
    if smiles_series is None:
        return pd.DataFrame(columns=["cid", "smiles", "mol_wt"])

    out["smiles"] = smiles_series
    if "MolecularWeight" in df.columns:
        out["mol_wt"] = pd.to_numeric(df["MolecularWeight"], errors="coerce")
    elif "mol_wt" in df.columns:
        out["mol_wt"] = pd.to_numeric(df["mol_wt"], errors="coerce")
    else:
        out["mol_wt"] = np.nan

    df = out[["cid", "smiles", "mol_wt"]].dropna(subset=["smiles"])
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df = df[df["smiles"].astype(bool)]
    df = df.drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    return df


@lru_cache(maxsize=128)
def _fetch_smiles_by_smarts_cached(
    smarts: str,
    max_cids: int = 5000,
    timeout: int = 30,
    retries: int = 2,
    property_workers: int = 4,
) -> pd.DataFrame:
    return _fetch_smiles_by_smarts_uncached(
        smarts=smarts,
        max_cids=max_cids,
        timeout=timeout,
        retries=retries,
        property_workers=property_workers,
    )


def fetch_smiles_by_smarts(
    smarts: str,
    max_cids: int = 5000,
    timeout: int = 30,
    retries: int = 2,
    property_workers: int = 4,
) -> pd.DataFrame:
    smarts = _clean_query_text(smarts)
    if not smarts:
        return pd.DataFrame(columns=["cid", "smiles", "mol_wt"])
    disk_cached_df = _load_smiles_cache(smarts, int(max_cids))
    if disk_cached_df is not None:
        validated_df = _filter_pubchem_query_matches(disk_cached_df, smarts)
        if len(validated_df) != len(disk_cached_df):
            _save_smiles_cache(smarts, int(max_cids), validated_df)
        validated_df.attrs["cache_source"] = "disk"
        return validated_df.copy(deep=True)
    worker_count = _normalize_worker_count(property_workers, max(1, int(max_cids) // 100 + 1), cap=6)
    df = _fetch_smiles_by_smarts_cached(
        smarts=smarts,
        max_cids=int(max_cids),
        timeout=int(timeout),
        retries=int(retries),
        property_workers=worker_count,
    )
    validated_df = _filter_pubchem_query_matches(df, smarts)
    _save_smiles_cache(smarts, int(max_cids), validated_df)
    validated_df.attrs["cache_source"] = "network"
    return validated_df.copy(deep=True)
