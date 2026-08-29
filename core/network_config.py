# -*- coding: utf-8 -*-
"""统一配置系统的网络访问代理。

代理配置同时支持环境变量和本地配置文件。环境变量适合部署场景，
本地配置文件则由 Streamlit 侧边栏维护，保存后立即作用于当前进程。
"""

from __future__ import annotations

import base64
import json
import os
import socket
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlsplit


DEFAULT_PROXY_URL = 'socks5://127.0.0.1:10808'
DEFAULT_PROXY_TYPE = 'SOCKS5'
_SCHEME_TO_TYPE = {
    'socks5': 'SOCKS5',
    'socks5h': 'SOCKS5',
    'socks4': 'SOCKS5',
    'socks4a': 'SOCKS5',
    'http': 'HTTP',
    'https': 'HTTPS',
}
_TYPE_TO_SCHEME = {
    'SOCKS5': 'socks5',
    'HTTP': 'http',
    'HTTPS': 'https',
}

NETWORK_CONFIG_PATH = Path(
    os.environ.get(
        'CFRP_NETWORK_CONFIG_PATH',
        str(Path(__file__).resolve().parent.parent / 'cache' / 'network_config.json'),
    )
)


def _read_saved_settings() -> Dict[str, Any]:
    try:
        if not NETWORK_CONFIG_PATH.is_file():
            return {}
        with NETWORK_CONFIG_PATH.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _env_value(name: str) -> Optional[str]:
    value = os.environ.get(name)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _parse_proxy_type(proxy_url: str) -> str:
    try:
        scheme = (urlsplit(str(proxy_url)).scheme or '').lower()
    except Exception:
        scheme = ''
    return _SCHEME_TO_TYPE.get(scheme, DEFAULT_PROXY_TYPE)


def normalize_proxy_url(proxy_url: str, proxy_type: Optional[str] = None) -> str:
    """校验并规范化代理地址；SOCKS5 内部使用 socks5h 解析 DNS。"""
    value = str(proxy_url or '').strip()
    if not value:
        raise ValueError('代理地址不能为空')

    selected_type = str(proxy_type or '').strip().upper()
    if selected_type not in _TYPE_TO_SCHEME:
        selected_type = _parse_proxy_type(value)

    parsed = urlsplit(value if '://' in value else f'{_TYPE_TO_SCHEME[selected_type]}://{value}')
    if not parsed.hostname:
        raise ValueError('代理地址缺少主机名，例如 127.0.0.1:10808')
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError('代理端口不是有效数字') from exc
    if port is None or not 1 <= int(port) <= 65535:
        raise ValueError('代理端口必须位于 1-65535')

    scheme = _TYPE_TO_SCHEME[selected_type]
    auth = ''
    if parsed.username is not None:
        password = parsed.password or ''
        auth = f'{parsed.username}:{password}@'
    return f'{scheme}://{auth}{parsed.hostname}:{int(port)}'


def _runtime_proxy_url(proxy_url: str) -> str:
    parsed = urlsplit(proxy_url)
    scheme = (parsed.scheme or '').lower()
    if scheme == 'socks5':
        return 'socks5h://' + proxy_url.split('://', 1)[1]
    if scheme == 'socks4':
        return 'socks4a://' + proxy_url.split('://', 1)[1]
    return proxy_url


def get_network_settings() -> Dict[str, Any]:
    """返回当前生效的代理设置，环境变量优先于本地配置文件。"""
    saved = _read_saved_settings()
    enabled_value = _env_value('CFRP_PROXY_ENABLED')
    url_value = _env_value('CFRP_PROXY_URL')
    has_saved_settings = bool(saved)
    enabled = saved.get('enabled', True) if has_saved_settings else True
    if not has_saved_settings and enabled_value is not None:
        enabled = enabled_value.lower() not in {'0', 'false', 'no', 'off'}
    proxy_url = (
        saved.get('proxy_url')
        if has_saved_settings
        else (url_value or DEFAULT_PROXY_URL)
    )
    proxy_type = str(saved.get('proxy_type') or _parse_proxy_type(proxy_url)).upper()
    if proxy_type not in _TYPE_TO_SCHEME:
        proxy_type = _parse_proxy_type(proxy_url)
    try:
        display_url = normalize_proxy_url(proxy_url, proxy_type)
    except ValueError:
        display_url = str(proxy_url).strip()
    return {
        'enabled': bool(enabled),
        'proxy_url': display_url,
        'proxy_type': proxy_type,
        'config_path': str(NETWORK_CONFIG_PATH),
        'source': 'file' if has_saved_settings else (
            'environment' if enabled_value is not None or url_value is not None else 'default'
        ),
    }


def apply_network_settings(settings: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """把设置应用到当前进程；禁用时清除代理环境变量。"""
    current = settings or get_network_settings()
    proxy = None
    if bool(current.get('enabled', True)):
        try:
            proxy = _runtime_proxy_url(
                normalize_proxy_url(current.get('proxy_url', DEFAULT_PROXY_URL), current.get('proxy_type'))
            )
        except ValueError:
            proxy = None

    names = ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')
    if proxy:
        for name in names:
            os.environ[name] = proxy
    else:
        for name in names:
            os.environ.pop(name, None)
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
    os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '120')
    return proxy


def save_network_settings(
    enabled: bool,
    proxy_url: str,
    proxy_type: Optional[str] = None,
) -> Dict[str, Any]:
    """保存 UI 配置并返回规范化后的设置。"""
    normalized_url = normalize_proxy_url(proxy_url, proxy_type)
    normalized_type = str(proxy_type or _parse_proxy_type(normalized_url)).upper()
    from datetime import datetime

    payload = {
        'enabled': bool(enabled),
        'proxy_url': normalized_url,
        'proxy_type': normalized_type,
        'updated_at': datetime.now().isoformat(timespec='seconds'),
    }
    NETWORK_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = NETWORK_CONFIG_PATH.with_suffix('.tmp')
    with temporary_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary_path, NETWORK_CONFIG_PATH)
    apply_network_settings(payload)
    return get_network_settings()


def get_proxy_url() -> Optional[str]:
    """返回 requests/Hugging Face 使用的代理地址。"""
    settings = get_network_settings()
    if not settings['enabled']:
        return None
    try:
        return _runtime_proxy_url(normalize_proxy_url(settings['proxy_url'], settings['proxy_type']))
    except ValueError:
        return None


def get_proxy_dict() -> Dict[str, str]:
    """返回 requests/huggingface_hub 可直接使用的代理字典。"""
    proxy = get_proxy_url()
    if not proxy:
        return {}
    return {'http': proxy, 'https': proxy}


def configure_network_proxy() -> Optional[str]:
    """兼容旧调用：读取并应用统一代理环境变量。"""
    return apply_network_settings()


def _decode_proxy_reply(reply: bytes) -> str:
    return bytes(reply or b'').decode('utf-8', errors='replace')


def _proxy_reply_contains_access_denied(reply: bytes) -> bool:
    text = _decode_proxy_reply(reply).lower()
    return any(
        marker in text
        for marker in (
            'unauthorized',
            'forbidden',
            'access denied',
            '禁止外部用户',
            '禁止外部用戶',
            'auth result',
        )
    )


def _proxy_authorization_header(proxy_url: str) -> Optional[str]:
    """Build a Basic proxy-auth header without exposing credentials in reports."""
    parsed = urlsplit(proxy_url)
    if parsed.username is None:
        return None
    username = unquote(parsed.username)
    password = unquote(parsed.password or '')
    token = base64.b64encode(f'{username}:{password}'.encode('utf-8')).decode('ascii')
    return f'Proxy-Authorization: Basic {token}'


def _probe_socks5_endpoint(proxy_url: str, timeout: float = 5.0) -> Dict[str, Any]:
    """Probe a SOCKS5 endpoint and classify common HTTP-port mistakes."""
    parsed = urlsplit(proxy_url)
    result = {'status': 'unknown', 'message': '', 'detected_type': None}
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=timeout) as sock:
            sock.sendall(b'\x05\x01\x00')
            reply = sock.recv(256)
        if reply.startswith((b'HTTP/', b'ICY ')) or b'server: proxy' in reply.lower():
            result.update(
                status='protocol_mismatch',
                detected_type='HTTP',
                message=(
                    '该端口返回 HTTP 代理响应，不是 SOCKS5。请将代理类型改为 HTTP；'
                    '如果仍提示禁止外部用户，请检查代理软件的认证或出口权限。'
                ),
            )
        elif len(reply) >= 2 and reply[0] == 5:
            if reply[1] == 0xFF:
                result.update(
                    status='failed',
                    message='SOCKS5 服务器拒绝所有认证方式，请检查代理认证配置。',
                )
            else:
                result.update(status='ok', message='SOCKS5 握手成功。')
        else:
            result.update(status='failed', message='未收到有效的 SOCKS5 握手响应。')
    except Exception as exc:
        result.update(status='failed', message=f'SOCKS5 握手失败：{exc}')
    return result


def _probe_http_proxy_endpoint(proxy_url: str, timeout: float = 5.0) -> Dict[str, Any]:
    """Probe an HTTP CONNECT proxy without issuing a full TLS request."""
    parsed = urlsplit(proxy_url)
    result = {'status': 'unknown', 'message': ''}
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=timeout) as sock:
            target = 'pubchem.ncbi.nlm.nih.gov:443'
            request = (
                f'CONNECT {target} HTTP/1.1\r\n'
                f'Host: {target}\r\n'
                'Proxy-Connection: Keep-Alive\r\n'
            )
            authorization = _proxy_authorization_header(proxy_url)
            if authorization:
                request += f'{authorization}\r\n'
            request += '\r\n'
            sock.sendall(request.encode('ascii'))
            reply = sock.recv(2048)
        text = _decode_proxy_reply(reply)
        first_line = text.splitlines()[0] if text.splitlines() else ''
        if not first_line.upper().startswith('HTTP/'):
            result.update(status='failed', message='HTTP 代理未返回有效的 HTTP 响应。')
        else:
            parts = first_line.split()
            code = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            if code in (401, 407):
                result.update(status='auth_required', message='HTTP 代理要求认证，请填写正确的用户名和密码。')
            elif _proxy_reply_contains_access_denied(reply):
                result.update(
                    status='access_denied',
                    message='HTTP 代理端口可连接，但代理服务器拒绝外部用户或要求认证。',
                )
            elif code is not None and 200 <= code < 300:
                result.update(status='ok', message='HTTP CONNECT 代理握手成功。')
            else:
                result.update(status='failed', message=f'HTTP 代理返回状态：{first_line}')
    except Exception as exc:
        result.update(status='failed', message=f'HTTP 代理握手失败：{exc}')
    return result

def test_network_connections(
    timeout: int = 10,
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """测试给定或当前代理对 PubChem 与 Hugging Face 的访问。"""
    settings = dict(settings or get_network_settings())
    settings.setdefault('proxy_type', _parse_proxy_type(settings.get('proxy_url', DEFAULT_PROXY_URL)))
    result: Dict[str, Any] = {'enabled': bool(settings['enabled']), 'settings': settings}
    if not settings['enabled']:
        result.update(status='disabled', message='代理未启用，未执行联网测试。')
        return result

    try:
        proxy = _runtime_proxy_url(
            normalize_proxy_url(settings.get('proxy_url', DEFAULT_PROXY_URL), settings.get('proxy_type'))
        )
    except ValueError as exc:
        result.update(status='failed', message=f'代理地址无效：{exc}')
        return result
    if settings['proxy_type'] == 'SOCKS5':
        result['protocol'] = _probe_socks5_endpoint(proxy, timeout=min(timeout, 8))
        if result['protocol'].get('status') != 'ok':
            http_proxy = proxy.replace('socks5h://', 'http://', 1).replace('socks5://', 'http://', 1)
            http_probe = _probe_http_proxy_endpoint(
                http_proxy,
                timeout=min(timeout, 8),
            )
            result['protocol']['http_probe'] = http_probe
            if http_probe.get('status') in {'ok', 'access_denied', 'auth_required'}:
                result['protocol'].update(
                    status='protocol_mismatch',
                    detected_type='HTTP',
                    message=(
                        '该端口实际提供 HTTP 代理，不是 SOCKS5。请将代理类型改为 HTTP；'
                        '如果 HTTP 诊断提示拒绝访问，请检查代理软件的认证或出口权限。'
                    ),
                )
                result['status'] = 'protocol_mismatch'
                result['message'] = result['protocol']['message']
            else:
                result['status'] = result['protocol'].get('status', 'failed')
                result['message'] = result['protocol'].get('message', 'SOCKS5 代理握手失败。')
            return result
    elif settings['proxy_type'] == 'HTTP':
        result['protocol'] = _probe_http_proxy_endpoint(proxy, timeout=min(timeout, 8))
        if result['protocol'].get('status') != 'ok':
            result['status'] = result['protocol'].get('status', 'failed')
            result['message'] = result['protocol'].get('message', 'HTTP 代理握手失败。')
            return result

    try:
        import requests
    except Exception as exc:
        result.update(status='failed', message=f'缺少 requests 或 SOCKS 支持：{exc}')
        return result

    endpoints = {
        'pubchem': 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/property/IsomericSMILES/JSON',
        'huggingface': 'https://huggingface.co/ibm/MoLFormer-XL-both-10pct/resolve/main/config.json',
    }
    for name, url in endpoints.items():
        try:
            response = requests.get(
                url,
                headers={'User-Agent': 'ML-CFRP-HTVS/1.0'},
                proxies={'http': proxy, 'https': proxy} if proxy else None,
                timeout=max(1, int(timeout)),
            )
            response.raise_for_status()
            result[name] = {'status': 'ok', 'code': int(response.status_code), 'message': '连接成功。'}
        except Exception as exc:
            result[name] = {'status': 'failed', 'message': str(exc)}
    successful = [result[name].get('status') == 'ok' for name in endpoints if name in result]
    result['status'] = 'ok' if successful and all(successful) else 'partial_or_failed'
    result['message'] = '测试完成。'
    return result


NETWORK_PROXY = configure_network_proxy()
