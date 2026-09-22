"""AI 输入助手的响应缓存。

设计依据：docs/superpowers/specs/2026-09-22-portal-formulation-first-input-design.md §4.5

动机
----
用户反复微调配方时会重复提交相同文本（"E-51/DDS，100:33，180度4小时"），
每次都调用 AI 既慢又费额度。命中缓存即复用。

键
--
``sha256(service_id | model | prompt_kind | 规范化输入文本)``

不同 service / model / prompt_kind **不得串味**——否则会拿到别的模型或别的
任务类型的结果，产生难以察觉的错误。

输入规范化
----------
去首尾空白 + 压缩连续空白；**大小写敏感保留**（SMILES 的 ``c1ccccc1`` 与
``C1CCCCC1`` 是不同分子，大小写必须参与键）。

安全
----
不落 API key、不落完整敏感输入。落盘前统一走 ``_redact`` 脱敏，
与 ``core/portal_tasks.py`` 的口径一致。

存储
----
``<root>/ai_cache/<sha256>.json``，每文件一条；LRU 上限默认 500。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Mapping

__all__ = ["PortalAICache", "cache_key", "normalize_input_text", "redact_text"]

#: 默认 LRU 上限
DEFAULT_MAX_ENTRIES = 500

#: 缓存子目录名
CACHE_DIRNAME = "ai_cache"

#: 脱敏：key=value / key: value 形式
_SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password|passwd"
    r"|authorization|x-api-key)(\s*[:=]\s*|\s+)([\"']?)([^\s,;}&\"']+)\3?"
)
#: 脱敏：Bearer <token>
_BEARER_PATTERN = re.compile(r"(?i)(\bbearer\s+)[^\s,;}&]+")
#: 脱敏：形如 sk-xxxx 的裸密钥
_BARE_KEY_PATTERN = re.compile(r"\b(sk|pk|rk)-[A-Za-z0-9_\-]{8,}\b")


def redact_text(text: str) -> str:
    """对文本脱敏（API key / token / bearer 一律替换为 [redacted]）。"""
    value = str(text or "")
    value = _SECRET_PATTERN.sub(r"\1\2[redacted]", value)
    value = _BEARER_PATTERN.sub(r"\1[redacted]", value)
    value = _BARE_KEY_PATTERN.sub("[redacted]", value)
    return value


def normalize_input_text(text: str) -> str:
    """规范化输入文本：去首尾空白 + 压缩连续空白；保留大小写。"""
    return re.sub(r"\s+", " ", str(text or "").strip())


def cache_key(*, service_id: str, model: str, prompt_kind: str, text: str) -> str:
    """计算缓存键（sha256 十六进制）。

    分隔符用 ``\\x1f``（不可打印），避免不同字段拼接产生歧义键
    （如 service="a", model="b|c" 与 service="a|b", model="c"）。
    """
    material = "\x1f".join(
        (
            str(service_id or "").strip(),
            str(model or "").strip(),
            str(prompt_kind or "").strip(),
            normalize_input_text(text),
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class PortalAICache:
    """文件型 AI 响应缓存（LRU）。

    每条缓存一个 JSON 文件，便于并发写入时互不干扰、且可单独删除。
    """

    def __init__(self, *, root: str | Path, max_entries: int = DEFAULT_MAX_ENTRIES):
        self.root = Path(root).expanduser()
        self.max_entries = max(1, int(max_entries))
        self.dir = self.root / CACHE_DIRNAME

    # -- 内部 ---------------------------------------------------------------

    def _path_for(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def _read(self, path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _write(self, path: Path, payload: Mapping[str, Any]) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
        except OSError:
            # 缓存写入失败不应影响主流程
            pass

    def _entries(self) -> list[tuple[float, Path]]:
        try:
            files = list(self.dir.glob("*.json"))
        except OSError:
            return []
        entries: list[tuple[float, Path]] = []
        for path in files:
            try:
                entries.append((path.stat().st_mtime, path))
            except OSError:
                continue
        return entries

    def _evict_if_needed(self) -> None:
        entries = self._entries()
        overflow = len(entries) - self.max_entries
        if overflow <= 0:
            return
        # 按 mtime 升序 = 最久未使用在前
        for _, path in sorted(entries, key=lambda item: item[0])[:overflow]:
            try:
                path.unlink()
            except OSError:
                continue

    # -- 公开接口 -----------------------------------------------------------

    def get(
        self,
        *,
        service_id: str,
        model: str,
        prompt_kind: str,
        text: str,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        """读缓存；未命中/损坏/force_refresh 时返回 None。

        命中时刷新文件 mtime（实现 LRU 语义），并递增 ``hits``。
        """
        if force_refresh:
            return None
        key = cache_key(
            service_id=service_id, model=model, prompt_kind=prompt_kind, text=text
        )
        path = self._path_for(key)
        payload = self._read(path)
        if payload is None:
            return None
        hits = int(payload.get("hits") or 0) + 1
        payload["hits"] = hits
        payload["last_hit_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._write(path, payload)
        # 刷新 mtime 以体现"最近使用"
        try:
            path.touch()
        except OSError:
            pass
        return {
            "value": payload.get("value"),
            "hits": hits,
            "created_at": payload.get("created_at"),
            "key": key,
        }

    def put(
        self,
        *,
        service_id: str,
        model: str,
        prompt_kind: str,
        text: str,
        value: Any,
        api_key: str | None = None,
    ) -> str:
        """写缓存，返回键。

        ``api_key`` 参数仅用于让调用方显式声明"此处可能有密钥"；
        它**不会**被写入磁盘（落盘前统一脱敏）。
        """
        key = cache_key(
            service_id=service_id, model=model, prompt_kind=prompt_kind, text=text
        )
        payload = {
            "key": key,
            "service_id": redact_text(str(service_id or "")),
            "model": redact_text(str(model or "")),
            "prompt_kind": str(prompt_kind or ""),
            "input_digest": hashlib.sha256(
                normalize_input_text(text).encode("utf-8")
            ).hexdigest(),
            "value": value,
            "hits": 0,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._write(self._path_for(key), payload)
        self._evict_if_needed()
        return key

    def size(self) -> int:
        """当前缓存条目数。"""
        return len(self._entries())

    def clear(self) -> None:
        """清空全部缓存。"""
        for _, path in self._entries():
            try:
                path.unlink()
            except OSError:
                continue
