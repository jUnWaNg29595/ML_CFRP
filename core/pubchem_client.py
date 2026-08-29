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
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

try:
    import requests
except Exception:
    requests = None

from .network_config import configure_network_proxy, get_proxy_dict

configure_network_proxy()

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    RDKIT_AVAILABLE = True
except Exception:
    Chem = None
    Descriptors = None
    RDKIT_AVAILABLE = False

import numpy as np
import pandas as pd

from .melting_point_data import canonicalize_smiles, parse_melting_point_text

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_VIEW_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
PUBCHEM_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "pubchem"


def _notify_progress(callback: Optional[Callable[[Dict], None]], payload: Dict) -> None:
    if not callable(callback):
        return
    try:
        callback(dict(payload))
    except Exception:
        return


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
    if requests is not None:
        last_err = None
        method = 'POST' if data else 'GET'
        for attempt in range(max(1, int(retries) + 1)):
            try:
                response = requests.request(
                    method,
                    url,
                    data=data,
                    headers={
                        'User-Agent': 'ML-CFRP-HTVS/1.0',
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    timeout=timeout,
                    proxies=get_proxy_dict() or None,
                )
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError, OSError) as exc:
                last_err = exc
                status_code = getattr(getattr(exc, 'response', None), 'status_code', 0)
                if attempt < retries:
                    retry_after = 0.0
                    try:
                        retry_after = float(
                            getattr(getattr(exc, 'response', None), 'headers', {}).get(
                                'Retry-After', 0
                            ) or 0
                        )
                    except (AttributeError, TypeError, ValueError):
                        retry_after = 0.0
                    delay = float(backoff) * (2 ** attempt)
                    if status_code in {429, 500, 502, 503, 504}:
                        delay = max(delay, 2.0 * (2 ** attempt))
                    time.sleep(max(delay, retry_after))
                    continue
                raise RuntimeError(f'PubChem request failed: {last_err}') from exc
        raise RuntimeError(f'PubChem request failed: {last_err}')

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
                retry_after = 0.0
                if isinstance(exc, HTTPError):
                    try:
                        retry_after = float(exc.headers.get('Retry-After', 0) or 0)
                    except (AttributeError, TypeError, ValueError):
                        retry_after = 0.0
                server_busy = isinstance(exc, HTTPError) and getattr(exc, 'code', 0) in {
                    429,
                    500,
                    502,
                    503,
                    504,
                }
                delay = float(backoff) * (2 ** attempt)
                if server_busy:
                    delay = max(delay, 2.0 * (2 ** attempt))
                time.sleep(max(delay, retry_after))
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


def split_structure_queries(value: object) -> List[str]:
    """Split a user-provided structure query list into unique normalized entries."""
    if value is None:
        return []
    text = str(value).replace('\r', '\n')
    queries: List[str] = []
    seen = set()
    chunks: List[str] = []
    current: List[str] = []
    bracket_depth = 0
    for character in text:
        if character == '[':
            bracket_depth += 1
        elif character == ']' and bracket_depth:
            bracket_depth -= 1
        if character == '\n' or (character == ';' and bracket_depth == 0):
            chunks.append(''.join(current))
            current = []
        else:
            current.append(character)
    chunks.append(''.join(current))
    for item in chunks:
        query = _clean_query_text(item)
        if query and query not in seen:
            queries.append(query)
            seen.add(query)
    return queries


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
    try:
        max_cids = max(0, int(max_cids))
    except (TypeError, ValueError):
        max_cids = 0
    if not smarts or max_cids <= 0:
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
                f"{PUBCHEM_BASE}/compound/fastsubstructure/smarts/cids/JSON",
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
    try:
        max_cids = max(0, int(max_cids))
    except (TypeError, ValueError):
        max_cids = 0
    if max_cids <= 0:
        return []
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
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> pd.DataFrame:
    if not cids:
        return pd.DataFrame()
    ordered_cids = list(dict.fromkeys(int(cid) for cid in cids))
    _notify_progress(
        progress_callback,
        {
            "phase": "properties",
            "total_cids": len(ordered_cids),
            "completed_cids": 0,
            "cache_hits": 0,
            "fetched_cids": 0,
            "failed_cids": 0,
        },
    )
    props = ",".join([p for p in properties if p])
    chunks = list(_chunked(ordered_cids, 100))
    rows = []
    completed_cids = 0

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
            completed_cids += len(chunk)
            _notify_progress(
                progress_callback,
                {
                    "phase": "properties",
                    "total_cids": len(ordered_cids),
                    "completed_cids": completed_cids,
                    "cache_hits": 0,
                    "fetched_cids": completed_cids,
                    "failed_cids": 0,
                },
            )
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(_fetch_chunk, chunk): idx
                for idx, chunk in enumerate(chunks)
            }
            chunk_tables: List[Tuple[int, List[Dict]]] = []
            for future in as_completed(future_map):
                idx = future_map[future]
                chunk = chunks[idx]
                table = future.result()
                if table:
                    chunk_tables.append((idx, table))
                completed_cids += len(chunk)
                _notify_progress(
                    progress_callback,
                    {
                        "phase": "properties",
                        "total_cids": len(ordered_cids),
                        "completed_cids": completed_cids,
                        "cache_hits": 0,
                        "fetched_cids": completed_cids,
                        "failed_cids": 0,
                    },
                )
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
        properties=["CanonicalSMILES", "ConnectivitySMILES", "IsomericSMILES", "SMILES", "MolecularWeight"],
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
            cand = df[col].astype("string")
            cand = cand.mask(cand.str.strip().eq(""), pd.NA)
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


_MELTING_POINT_CLIENT_SCHEMA_VERSION = 1
_MELTING_POINT_ANNOTATION_COLUMNS = [
    "cid", "mp_raw", "source_url", "source_name", "source_record"
]


def _melting_point_cache_path(kind: str, payload: Dict) -> Path:
    cache_key = dict(payload)
    cache_key["schema_version"] = _MELTING_POINT_CLIENT_SCHEMA_VERSION
    digest = hashlib.sha256(
        json.dumps(cache_key, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return PUBCHEM_CACHE_DIR / f"melting_point_{kind}_{digest}.csv"


def _melting_point_cid_cache_path(cid: int) -> Path:
    return PUBCHEM_CACHE_DIR / (
        f"melting_point_annotation_cid_{int(cid)}_v{_MELTING_POINT_CLIENT_SCHEMA_VERSION}.json"
    )


def _load_melting_point_cid_cache(cid: int) -> tuple[bool, List[Dict]]:
    path = _melting_point_cid_cache_path(cid)
    if not path.exists():
        return False, []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows", []) if isinstance(payload, dict) else payload
        return True, rows if isinstance(rows, list) else []
    except Exception:
        return False, []


def _save_melting_point_cid_cache(cid: int, rows: List[Dict]) -> None:
    PUBCHEM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _melting_point_cid_cache_path(cid)
    path.write_text(
        json.dumps(
            {"schema_version": _MELTING_POINT_CLIENT_SCHEMA_VERSION, "cid": int(cid), "rows": rows},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _annotation_value_strings(value: object) -> List[str]:
    if not isinstance(value, dict):
        return []
    strings = []
    markup = value.get("StringWithMarkup")
    if isinstance(markup, list):
        strings.extend(str(item.get("String", "")).strip() for item in markup if isinstance(item, dict))
    elif markup:
        strings.append(str(markup).strip())
    if value.get("String") is not None:
        strings.append(str(value["String"]).strip())
    number = value.get("Number")
    if number is not None:
        numbers = number if isinstance(number, list) else [number]
        unit = value.get("Unit", "")
        suffix = f" {unit}" if unit else ""
        strings.extend(f"{item}{suffix}".strip() for item in numbers)
    return [item for item in strings if item]


def _extract_melting_point_annotations(cid: int, payload: Dict) -> List[Dict]:
    record = payload.get("Record", {}) if isinstance(payload, dict) else {}
    source_record = record.get("RecordNumber", cid)
    record_references = {}
    for reference in record.get("Reference", []) or []:
        if isinstance(reference, dict):
            number = reference.get("ReferenceNumber")
            if number is not None:
                record_references[str(number)] = reference
    rows = []

    def reference_rows(value):
        if isinstance(value, dict):
            return [value]
        if isinstance(value, (list, tuple)):
            return list(value)
        if value is None:
            return []
        return [value]

    def reference_metadata(information):
        refs = reference_rows(information.get("Reference"))
        if not refs:
            refs = reference_rows(information.get("References"))
        if not refs:
            return '', ''
        for item in refs:
            if isinstance(item, dict):
                reference = item
            else:
                reference = record_references.get(str(item), {})
            if not isinstance(reference, dict):
                continue
            source_url = reference.get("URL") or reference.get("SourceURL") or reference.get("Url") or ''
            source_name = (
                reference.get("SourceName")
                or reference.get("Name")
                or reference.get("Source")
                or ''
            )
            if source_url or source_name:
                return str(source_url), str(source_name)
        return '', ''

    def walk(section: object) -> None:
        if not isinstance(section, dict):
            return
        heading = str(section.get("TOCHeading", "")).strip().casefold()
        if heading == "melting point":
            for information in section.get("Information", []) or []:
                if not isinstance(information, dict):
                    continue
                values = _annotation_value_strings(information.get("Value", {}))
                source_url, source_name = reference_metadata(information)
                for raw in values:
                    rows.append({
                        "cid": int(cid),
                        "mp_raw": raw,
                        "source_url": source_url,
                        "source_name": source_name,
                        "source_record": source_record,
                    })
        for child in section.get("Section", []) or []:
            walk(child)

    for section in record.get("Section", []) or []:
        walk(section)
    unique = []
    seen = set()
    for row in rows:
        key = (row["cid"], row["mp_raw"], row["source_url"], row["source_name"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def fetch_melting_point_annotations_by_cids(
    cids: Sequence[int],
    *,
    max_workers: int = 4,
    timeout: int = 30,
    retries: int = 2,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> pd.DataFrame:
    ordered_cids = list(dict.fromkeys(int(cid) for cid in cids))
    empty = pd.DataFrame(columns=_MELTING_POINT_ANNOTATION_COLUMNS)
    if not ordered_cids:
        return empty
    total_cids = len(ordered_cids)
    _notify_progress(
        progress_callback,
        {
            "phase": "annotations",
            "total_cids": total_cids,
            "completed_cids": 0,
            "cache_hits": 0,
            "fetched_cids": 0,
            "failed_cids": 0,
        },
    )
    cache_path = _melting_point_cache_path(
        "annotations", {"cids": ordered_cids}
    )
    if cache_path.exists():
        try:
            cached = pd.read_csv(cache_path)
            if all(column in cached.columns for column in _MELTING_POINT_ANNOTATION_COLUMNS):
                result = cached[_MELTING_POINT_ANNOTATION_COLUMNS].copy()
                result.attrs.update(
                    {
                        "failed_cids": [],
                        "failure_messages": {},
                        "cache_hits": total_cids,
                        "fetched_cids": 0,
                    }
                )
                _notify_progress(
                    progress_callback,
                    {
                        "phase": "annotations",
                        "total_cids": total_cids,
                        "completed_cids": total_cids,
                        "cache_hits": total_cids,
                        "fetched_cids": 0,
                        "failed_cids": 0,
                    },
                )
                return result
        except Exception:
            pass

    def fetch_one(cid: int) -> List[Dict]:
        url = f"{PUBCHEM_VIEW_BASE}/data/compound/{cid}/JSON"
        payload = _request_json(url, timeout=timeout, retries=retries)
        return _extract_melting_point_annotations(cid, payload)

    worker_count = _normalize_worker_count(max_workers, len(ordered_cids), cap=6)
    results = {}
    missing = []
    cache_hits = 0
    for cid in ordered_cids:
        found, rows = _load_melting_point_cid_cache(cid)
        if found:
            results[cid] = rows
            cache_hits += 1
        else:
            missing.append(cid)

    _notify_progress(
        progress_callback,
        {
            "phase": "annotations",
            "total_cids": total_cids,
            "completed_cids": cache_hits,
            "cache_hits": cache_hits,
            "fetched_cids": 0,
            "failed_cids": 0,
        },
    )

    failures = {}

    def fetch_with_status(cid: int) -> tuple[int, List[Dict], Optional[str]]:
        try:
            return cid, fetch_one(cid), None
        except Exception as exc:
            return cid, [], str(exc)

    if worker_count == 1:
        completed_results = (fetch_with_status(cid) for cid in missing)
        for cid, rows, error in completed_results:
            if error:
                failures[int(cid)] = error
            else:
                results[int(cid)] = rows
                _save_melting_point_cid_cache(int(cid), rows)
            _notify_progress(
                progress_callback,
                {
                    "phase": "annotations",
                    "total_cids": total_cids,
                    "completed_cids": cache_hits + len(results) - cache_hits + len(failures),
                    "cache_hits": cache_hits,
                    "fetched_cids": len(results) - cache_hits,
                    "failed_cids": len(failures),
                },
            )
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(fetch_with_status, cid) for cid in missing]
            for future in as_completed(futures):
                cid, rows, error = future.result()
                if error:
                    failures[int(cid)] = error
                else:
                    results[int(cid)] = rows
                    _save_melting_point_cid_cache(int(cid), rows)
                _notify_progress(
                    progress_callback,
                    {
                        "phase": "annotations",
                        "total_cids": total_cids,
                        "completed_cids": cache_hits + len(results) - cache_hits + len(failures),
                        "cache_hits": cache_hits,
                        "fetched_cids": len(results) - cache_hits,
                        "failed_cids": len(failures),
                    },
                )
    rows = [row for cid in ordered_cids for row in results.get(cid, [])]
    result = pd.DataFrame(rows, columns=_MELTING_POINT_ANNOTATION_COLUMNS)
    result.attrs.update(
        {
            "failed_cids": sorted(failures),
            "failure_messages": failures,
            "cache_hits": int(cache_hits),
            "fetched_cids": int(len(missing) - len(failures)),
        }
    )
    if not failures:
        PUBCHEM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return result


def fetch_melting_point_records_by_smarts(
    smarts: str,
    *,
    component_role: str,
    hardener_class: str = "",
    max_cids: int = 5000,
    property_workers: int = 4,
    timeout: int = 30,
    retries: int = 2,
    cids: Optional[Sequence[int]] = None,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> pd.DataFrame:
    query = _clean_query_text(smarts)
    try:
        max_cids = max(0, int(max_cids))
    except (TypeError, ValueError):
        max_cids = 0
    if not query or max_cids <= 0:
        return pd.DataFrame()
    cache_payload = {
        "query": query,
        "component_role": component_role,
        "hardener_class": hardener_class,
        "max_cids": int(max_cids),
    }
    cache_path = _melting_point_cache_path("records", cache_payload)
    if cache_path.exists():
        try:
            cached = pd.read_csv(cache_path)
            validated_cached = _filter_pubchem_query_matches(cached, query)
            validated_cached.attrs["cache_source"] = "disk"
            cached_cids = validated_cached["cid"].nunique() if "cid" in validated_cached.columns else 0
            validated_cached.attrs.update(
                {
                    "cids_requested": int(cached_cids),
                    "cache_hits": int(cached_cids),
                    "fetched_cids": 0,
                    "failed_cids": [],
                }
            )
            _notify_progress(
                progress_callback,
                {
                    "phase": "records",
                    "total_cids": int(cached_cids),
                    "completed_cids": int(cached_cids),
                    "cache_hits": int(cached_cids),
                    "fetched_cids": 0,
                    "failed_cids": 0,
                },
            )
            return validated_cached
        except Exception:
            pass
    requested_cids = (
        list(dict.fromkeys(int(cid) for cid in cids))
        if cids is not None
        else fetch_cids_by_smarts(query, max_cids=max_cids, timeout=timeout, retries=retries)
    )
    requested_cids = requested_cids[:max_cids]
    _notify_progress(
        progress_callback,
        {
            "phase": "properties",
            "total_cids": int(len(requested_cids)),
            "completed_cids": 0,
            "cache_hits": 0,
            "fetched_cids": 0,
            "failed_cids": 0,
        },
    )
    properties = fetch_properties_by_cids(
        requested_cids,
        properties=["CanonicalSMILES", "ConnectivitySMILES", "IsomericSMILES", "SMILES", "MolecularWeight"],
        timeout=timeout,
        retries=retries,
        max_workers=property_workers,
        progress_callback=progress_callback,
    )
    annotations = fetch_melting_point_annotations_by_cids(
        requested_cids,
        max_workers=property_workers,
        timeout=timeout,
        retries=retries,
        progress_callback=progress_callback,
    )
    if properties.empty or annotations.empty:
        return pd.DataFrame()
    properties = properties.rename(columns={"CID": "cid", "CanonicalSMILES": "smiles", "MolecularWeight": "mol_wt"})
    for column in ("cid", "smiles", "mol_wt"):
        if column not in properties:
            properties[column] = np.nan
    smiles_columns = [column for column in ("smiles", "ConnectivitySMILES", "IsomericSMILES", "SMILES") if column in properties]
    if smiles_columns:
        merged_smiles = properties[smiles_columns[0]].astype("string")
        merged_smiles = merged_smiles.mask(merged_smiles.str.strip().eq(""), pd.NA)
        for column in smiles_columns[1:]:
            fallback = properties[column].astype("string")
            fallback = fallback.mask(fallback.str.strip().eq(""), pd.NA)
            merged_smiles = merged_smiles.fillna(fallback)
        properties["smiles"] = merged_smiles
    properties["mol_wt"] = pd.to_numeric(properties["mol_wt"], errors="coerce")
    properties = _filter_pubchem_query_matches(properties, query)
    if properties.empty:
        return pd.DataFrame()
    result = annotations.merge(properties[["cid", "smiles", "mol_wt"]], on="cid", how="inner")
    if result.empty:
        return result
    parsed = result["mp_raw"].map(parse_melting_point_text).apply(pd.Series)
    for column in parsed.columns:
        result[column] = parsed[column]
    result["canonical_smiles"] = result["smiles"].map(canonicalize_smiles)
    result["component_role"] = component_role
    result["hardener_class"] = hardener_class
    result["source"] = result.get("source_name", pd.Series("PubChem PUG View", index=result.index)).fillna("PubChem PUG View")
    result["query_smarts"] = query
    result["max_cids"] = int(max_cids)
    result.attrs.update(annotations.attrs)
    result.attrs.update(
        {
            "query_validated": bool(properties.attrs.get("query_validated", False)),
            "query_raw_count": int(properties.attrs.get("raw_count", len(properties))),
            "query_rejected_count": int(properties.attrs.get("rejected_count", 0)),
        }
    )
    result.attrs["cids_requested"] = int(len(requested_cids))
    result.attrs["properties_fetched"] = int(len(properties))
    PUBCHEM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not annotations.attrs.get("failed_cids"):
        result.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return result


def fetch_melting_point_records_by_smarts_queries(
    queries: object,
    *,
    component_role: str = "unknown",
    hardener_class: str = "",
    per_query_max_cids: int = 5000,
    **kwargs,
) -> pd.DataFrame:
    """Fetch and merge records for multiple independent structure queries."""
    query_list = split_structure_queries(queries)
    frames: List[pd.DataFrame] = []
    for query in query_list:
        frame = fetch_melting_point_records_by_smarts(
            query,
            component_role=component_role,
            hardener_class=hardener_class,
            max_cids=per_query_max_cids,
            **kwargs,
        )
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frame = frame.copy()
            frame["query_group"] = query
            frames.append(frame)
    if not frames:
        result = pd.DataFrame()
        result.attrs.update({"query_count": len(query_list), "query_sources": query_list})
        return result
    result = pd.concat(frames, ignore_index=True)
    if "canonical_smiles" not in result.columns and "smiles" in result.columns:
        result["canonical_smiles"] = result["smiles"].map(canonicalize_smiles)
    dedup_columns = [column for column in ("canonical_smiles", "component_role", "hardener_class") if column in result.columns]
    if dedup_columns:
        result = result.drop_duplicates(subset=dedup_columns, keep="first").reset_index(drop=True)
    result.attrs.update({
        "query_count": len(query_list),
        "query_sources": query_list,
        "records_before_dedup": int(sum(len(frame) for frame in frames)),
        "records_after_dedup": int(len(result)),
    })
    return result


def _load_cached_properties_frames() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if not PUBCHEM_CACHE_DIR.exists():
        return pd.DataFrame(columns=["cid", "smiles", "mol_wt"])
    for path in PUBCHEM_CACHE_DIR.glob("*.csv"):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if not {"cid", "smiles"}.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["cid"] = pd.to_numeric(frame["cid"], errors="coerce")
        frame = frame[frame["cid"].notna()].copy()
        frame["smiles"] = frame["smiles"].fillna("").astype(str).str.strip()
        if "mol_wt" not in frame.columns:
            frame["mol_wt"] = np.nan
        frames.append(frame[["cid", "smiles", "mol_wt"]])
    if not frames:
        return pd.DataFrame(columns=["cid", "smiles", "mol_wt"])
    result = pd.concat(frames, ignore_index=True, sort=False)
    result["cid"] = result["cid"].astype(int)
    result = result[result["smiles"].ne("")].drop_duplicates("cid", keep="first")
    return result.reset_index(drop=True)


def _load_cached_annotation_rows() -> Dict[int, List[Dict]]:
    annotations: Dict[int, List[Dict]] = {}
    if not PUBCHEM_CACHE_DIR.exists():
        return annotations
    for path in PUBCHEM_CACHE_DIR.glob("melting_point_annotation_cid_*_v*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            payload = payload.get("rows", [])
        if not isinstance(payload, list):
            continue
        match = path.name.split("melting_point_annotation_cid_", 1)[-1].split("_v", 1)[0]
        try:
            cid = int(match)
        except (TypeError, ValueError):
            continue
        annotations[cid] = [row for row in payload if isinstance(row, dict)]
    return annotations


def build_cached_melting_point_records(
    queries: object,
    *,
    component_role: str = "resin",
    hardener_class: str = "",
    min_molecular_weight: float = 150.0,
    max_molecular_weight: Optional[float] = None,
    min_melting_point_c: float = -273.15,
    max_melting_point_c: float = 500.0,
    min_epoxy_count: int = 1,
    single_fragment_only: bool = True,
    exclude_cids: Optional[Sequence[int]] = None,
    exclude_canonical_smiles: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Build quality-filtered records from existing PubChem property/annotation caches."""
    if not RDKIT_AVAILABLE:
        return pd.DataFrame()
    query_molecules = []
    for query in split_structure_queries(queries):
        try:
            molecule = Chem.MolFromSmiles(query) if _is_plain_smiles_query(query) else Chem.MolFromSmarts(query)
        except Exception:
            molecule = None
        if molecule is not None:
            query_molecules.append(molecule)
    if not query_molecules:
        return pd.DataFrame()
    epoxy_pattern = None
    if int(min_epoxy_count or 0) > 0:
        try:
            epoxy_pattern = Chem.MolFromSmarts("C1OC1")
        except Exception:
            epoxy_pattern = None
    excluded_cids = {int(value) for value in (exclude_cids or [])}
    excluded_smiles = {str(value) for value in (exclude_canonical_smiles or []) if value}
    properties = _load_cached_properties_frames()
    annotations = _load_cached_annotation_rows()
    rows: List[Dict] = []
    seen_smiles = set(excluded_smiles)
    for _, property_row in properties.iterrows():
        cid = int(property_row["cid"])
        if cid in excluded_cids or cid not in annotations:
            continue
        smiles = str(property_row["smiles"]).strip()
        try:
            molecule = Chem.MolFromSmiles(smiles)
        except Exception:
            molecule = None
        if molecule is None:
            continue
        if single_fragment_only and len(Chem.GetMolFrags(molecule)) != 1:
            continue
        if not any(molecule.HasSubstructMatch(pattern) for pattern in query_molecules):
            continue
        epoxy_count = (
            len(molecule.GetSubstructMatches(epoxy_pattern, uniquify=True))
            if epoxy_pattern is not None
            else 0
        )
        if int(min_epoxy_count or 0) > epoxy_count:
            continue
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        if canonical in seen_smiles:
            continue
        molecular_weight = pd.to_numeric(property_row.get("mol_wt"), errors="coerce")
        if pd.isna(molecular_weight):
            molecular_weight = Descriptors.MolWt(molecule)
        if float(molecular_weight) < float(min_molecular_weight):
            continue
        if max_molecular_weight is not None and float(molecular_weight) > float(max_molecular_weight):
            continue
        for annotation in annotations[cid]:
            parsed = parse_melting_point_text(annotation.get("mp_raw"))
            mp_c = parsed.get("mp_c")
            if parsed.get("mp_quality") != "high" or mp_c is None:
                continue
            if not (float(min_melting_point_c) <= float(mp_c) <= float(max_melting_point_c)):
                continue
            seen_smiles.add(canonical)
            row = {
                "cid": cid,
                "smiles": smiles,
                "canonical_smiles": canonical,
                "mol_wt": float(molecular_weight),
                "component_role": component_role,
                "hardener_class": hardener_class,
                "source": "PubChem cached PUG View",
                "query_smarts": "; ".join(split_structure_queries(queries)),
                "n_epoxy": int(epoxy_count),
            }
            row.update(annotation)
            row.update(parsed)
            rows.append(row)
            break
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.drop_duplicates("canonical_smiles", keep="first").reset_index(drop=True)
    result.attrs.update({
        "cache_source": "disk",
        "candidate_count": int(len(result)),
        "min_molecular_weight": float(min_molecular_weight),
        "max_melting_point_c": float(max_melting_point_c),
    })
    return result
