import json

from core import network_config


def test_save_and_disable_proxy(monkeypatch, tmp_path):
    config_path = tmp_path / 'network_config.json'
    monkeypatch.setattr(network_config, 'NETWORK_CONFIG_PATH', config_path)
    monkeypatch.delenv('CFRP_PROXY_ENABLED', raising=False)
    monkeypatch.delenv('CFRP_PROXY_URL', raising=False)

    saved = network_config.save_network_settings(False, '127.0.0.1:8080', 'HTTP')

    assert saved['enabled'] is False
    assert network_config.get_proxy_url() is None
    assert json.loads(config_path.read_text(encoding='utf-8'))['proxy_type'] == 'HTTP'


def test_default_proxy_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(network_config, 'NETWORK_CONFIG_PATH', tmp_path / 'network_config.json')
    monkeypatch.delenv('CFRP_PROXY_ENABLED', raising=False)
    monkeypatch.delenv('CFRP_PROXY_URL', raising=False)

    settings = network_config.get_network_settings()

    assert settings['enabled'] is True
    assert settings['proxy_type'] == 'SOCKS5'
    assert settings['proxy_url'] == 'socks5://127.0.0.1:10808'


def test_proxy_url_normalization_and_runtime_dns(monkeypatch, tmp_path):
    monkeypatch.setattr(network_config, 'NETWORK_CONFIG_PATH', tmp_path / 'network_config.json')
    monkeypatch.delenv('CFRP_PROXY_ENABLED', raising=False)
    monkeypatch.delenv('CFRP_PROXY_URL', raising=False)

    network_config.save_network_settings(True, '127.0.0.1:10808', 'SOCKS5')

    assert network_config.get_proxy_url() == 'socks5h://127.0.0.1:10808'
    assert network_config.normalize_proxy_url('http://127.0.0.1:8080', 'HTTP') == 'http://127.0.0.1:8080'


def test_disabled_connection_test_does_not_request_network():
    report = network_config.test_network_connections(
        settings={
            'enabled': False,
            'proxy_type': 'HTTP',
            'proxy_url': '127.0.0.1:8080',
        }
    )

    assert report['status'] == 'disabled'


def test_socks5_probe_classifies_http_proxy(monkeypatch):
    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def sendall(self, _payload):
            return None

        def recv(self, _size):
            return b'HTTP/1.0 200 OK\\r\\nServer: Proxy\\r\\n\\r\\nUnauthorized ...'

    monkeypatch.setattr(
        network_config.socket,
        'create_connection',
        lambda *_args, **_kwargs: FakeSocket(),
    )

    report = network_config._probe_socks5_endpoint(
        'socks5h://127.0.0.1:10808',
        timeout=1,
    )

    assert report['status'] == 'protocol_mismatch'
    assert report['detected_type'] == 'HTTP'
    assert 'HTTP' in report['message']


def test_http_probe_classifies_external_access_denied(monkeypatch):
    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def sendall(self, _payload):
            return None

        def recv(self, _size):
            return (
                b'HTTP/1.0 200 OK\\r\\nServer: Proxy\\r\\n\\r\\n'
                b'Auth Result: \\xe7\\xa6\\x81\\xe6\\xad\\xa2\\xe5\\xa4\\x96\\xe9\\x83\\xa8\\xe7\\x94\\xa8\\xe6\\x88\\xb7\\xe3\\x80\\x82'
            )

    monkeypatch.setattr(
        network_config.socket,
        'create_connection',
        lambda *_args, **_kwargs: FakeSocket(),
    )

    report = network_config._probe_http_proxy_endpoint(
        'http://127.0.0.1:10808',
        timeout=1,
    )

    assert report['status'] == 'access_denied'
    assert '外部' in report['message']

def test_http_probe_sends_basic_proxy_auth(monkeypatch):
    captured = {}

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def sendall(self, payload):
            captured['payload'] = payload

        def recv(self, _size):
            return b'HTTP/1.1 200 Connection Established\\r\\n\\r\\n'

    monkeypatch.setattr(
        network_config.socket,
        'create_connection',
        lambda *_args, **_kwargs: FakeSocket(),
    )

    report = network_config._probe_http_proxy_endpoint(
        'http://user:pass@127.0.0.1:8080',
        timeout=1,
    )

    assert report['status'] == 'ok'
    assert b'Proxy-Authorization: Basic dXNlcjpwYXNz' in captured['payload']


def test_http_probe_prefers_auth_status_over_access_denied_body(monkeypatch):
    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def sendall(self, _payload):
            return None

        def recv(self, _size):
            return b'HTTP/1.1 407 Proxy Authentication Required\\r\\n\\r\\nAuth Result: Unauthorized'

    monkeypatch.setattr(
        network_config.socket,
        'create_connection',
        lambda *_args, **_kwargs: FakeSocket(),
    )

    report = network_config._probe_http_proxy_endpoint(
        'http://127.0.0.1:8080',
        timeout=1,
    )

    assert report['status'] == 'auth_required'


def test_http_connection_test_stops_before_online_requests(monkeypatch):
    monkeypatch.setattr(
        network_config,
        '_probe_http_proxy_endpoint',
        lambda *_args, **_kwargs: {
            'status': 'access_denied',
            'message': 'HTTP 代理端口可连接，但代理服务器拒绝外部用户或要求认证。',
        },
    )

    report = network_config.test_network_connections(
        settings={
            'enabled': True,
            'proxy_type': 'HTTP',
            'proxy_url': '127.0.0.1:10808',
        }
    )

    assert report['status'] == 'access_denied'
    assert 'pubchem' not in report
    assert 'huggingface' not in report


def test_socks5_probe_failure_stops_before_online_requests(monkeypatch):
    import sys
    from types import SimpleNamespace

    fake_requests = SimpleNamespace(
        get=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError('协议探测失败后不应访问在线服务')
        )
    )
    monkeypatch.setitem(sys.modules, 'requests', fake_requests)

    monkeypatch.setattr(
        network_config,
        '_probe_socks5_endpoint',
        lambda *_args, **_kwargs: {
            'status': 'failed',
            'message': 'SOCKS5 握手失败：连接被本机软件中止。',
        },
    )
    monkeypatch.setattr(
        network_config,
        '_probe_http_proxy_endpoint',
        lambda *_args, **_kwargs: {
            'status': 'failed',
            'message': 'HTTP 代理也未返回有效响应。',
        },
    )
    report = network_config.test_network_connections(
        settings={
            'enabled': True,
            'proxy_type': 'SOCKS5',
            'proxy_url': '127.0.0.1:10808',
        }
    )

    assert report['status'] == 'failed'
    assert 'pubchem' not in report
    assert 'huggingface' not in report


def test_socks5_probe_connection_abort_falls_back_to_http_diagnosis(monkeypatch):
    monkeypatch.setattr(
        network_config,
        '_probe_socks5_endpoint',
        lambda *_args, **_kwargs: {
            'status': 'failed',
            'message': 'SOCKS5 握手失败：WinError 10053。',
        },
    )
    monkeypatch.setattr(
        network_config,
        '_probe_http_proxy_endpoint',
        lambda *_args, **_kwargs: {
            'status': 'access_denied',
            'message': 'HTTP 代理端口可连接，但代理服务器拒绝外部用户或要求认证。',
        },
    )

    report = network_config.test_network_connections(
        settings={
            'enabled': True,
            'proxy_type': 'SOCKS5',
            'proxy_url': '127.0.0.1:10808',
        }
    )

    assert report['status'] == 'protocol_mismatch'
    assert report['protocol']['detected_type'] == 'HTTP'
    assert report['protocol']['http_probe']['status'] == 'access_denied'
    assert 'pubchem' not in report
    assert 'huggingface' not in report
