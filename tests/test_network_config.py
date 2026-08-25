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
