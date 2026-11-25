from collections import namedtuple
from threading import Event, Thread, enumerate as running_threads
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tts_audiobook_tool import web_ollama_progress as progress


Address = namedtuple("Address", ["ip", "port"])


def listener(ip, port, pid=None):
    return SimpleNamespace(status=progress.psutil.CONN_LISTEN, laddr=Address(ip, port), pid=pid)


@pytest.mark.parametrize("binary,command,expected", [
    ("ollama", ["ollama", "serve"], True),
    ("ollama", ["ollama", "run"], False),
    ("another-server", ["ollama", "serve"], False),
])
def test_discovery_requires_the_local_ollama_server_and_its_loopback_worker(monkeypatch, binary, command, expected):
    server = MagicMock()
    server.exe.return_value = "/bin/" + binary
    server.cmdline.return_value = command
    monkeypatch.setattr(progress.psutil, "net_connections", lambda **kw: [
        listener("0.0.0.0", 11434, 90), listener("127.0.0.1", 11433, 91), listener("127.0.0.1", 11434, 92),
    ])

    def process(pid):
        assert pid == 92
        return server

    monkeypatch.setattr(progress.psutil, "Process", process)
    assert progress._ollama_server() is (server if expected else None)
    runner, unrelated = MagicMock(), MagicMock()
    runner.exe.return_value = "/opt/ollama/llama-server"
    runner.net_connections.return_value = [listener("0.0.0.0", 80), listener("127.0.0.1", 39995)]
    unrelated.exe.return_value = "/bin/other"
    server.children.return_value = [unrelated, runner]
    assert progress._runner_ports(server) == [(runner, 39995)]


def slot(seed, done, **fields):
    return dict(is_processing=True, params={"seed": seed}, id_task=7, n_prompt_tokens=99,
                n_prompt_tokens_processed=done, **fields)


def test_slot_only_reports_completed_numeric_work_for_matching_request():
    http = MagicMock()
    response = http.get.return_value.__enter__.return_value
    response.status_code = 200
    response.json.return_value = [
        slot(987, 82), dict(slot(123, 111), is_processing=False), slot(123, "partial"), slot(123, 71),
    ]
    assert progress._read_slot(http, 31111, 123) == (7, 71)
    http.get.assert_called_once_with("http://127.0.0.1:31111/slots", allow_redirects=False, timeout=0.5)
    assert progress._read_slot(http, 31111, 456) is None
    response.json.return_value.append(slot(123, 63))
    assert progress._read_slot(http, 31111, 123) is None
    response.status_code = 307
    assert progress._read_slot(http, 31111, 987) is None


def test_observer_emits_only_advancing_counts_and_stops_before_generation(monkeypatch):
    server, runner = MagicMock(), MagicMock()
    monkeypatch.setattr(progress, "_POLL_SECONDS", 0.001)
    monkeypatch.setattr(progress, "_ollama_server", lambda: server)
    discovery = MagicMock(return_value=[(runner, 31000)])
    monkeypatch.setattr(progress, "_runner_ports", discovery)
    readings = iter([None, (12, 0), (12, 31), (12, 31), (12, 2048)])
    seen = []
    reported = Event()

    def read(http, port, seed):
        assert http.trust_env is False and seed == 456 and port == 31000
        return next(readings, (12, 2048))

    def report(detail):
        seen.append(detail)
        if len(seen) == 2:
            reported.set()

    monkeypatch.setattr(progress, "_read_slot", read)
    with progress.observe_prompt_reading(456, report) as stop:
        assert reported.wait(2)
        stop()
        assert not any(t.name == "qwen-prompt-progress" for t in running_threads())
    assert seen == ["Qwen is reading the prompt · 31 input tokens processed",
                    "Qwen is reading the prompt · 2,048 input tokens processed"]
    discovery.assert_called_once_with(server)


def test_stop_suppresses_late_result_and_joins_the_worker(monkeypatch):
    entered, release, progress_event = Event(), Event(), Event()
    monkeypatch.setattr(progress, "Event", lambda: progress_event)
    seen = []
    monkeypatch.setattr(progress, "_POLL_SECONDS", 0.001)
    monkeypatch.setattr(progress, "_ollama_server", lambda: MagicMock())
    monkeypatch.setattr(progress, "_runner_ports", lambda _server: [(MagicMock(), 31777)])

    def read(*_args):
        entered.set()
        release.wait(2)
        return (3, 199)

    monkeypatch.setattr(progress, "_read_slot", read)
    with progress.observe_prompt_reading(109, seen.append) as stop:
        assert entered.wait(2)

        waiter = Thread(target=stop)
        waiter.start()
        try:
            # Wait until stop() has set the polling event before releasing the HTTP stub.
            assert progress_event.wait(2)
        finally:
            release.set()
            waiter.join(2)
        assert not waiter.is_alive()
    assert seen == []


def test_missing_permissions_or_disabled_observer_do_not_interrupt_analysis(monkeypatch):
    checked = Event()

    def unavailable():
        checked.set()
        raise progress.psutil.AccessDenied(3)

    monkeypatch.setattr(progress, "_POLL_SECONDS", 0.001)
    monkeypatch.setattr(progress, "_ollama_server", unavailable)
    with progress.observe_prompt_reading(123, None) as stop:
        stop()
    assert not checked.is_set()
    with progress.observe_prompt_reading(123, pytest.fail):
        assert checked.wait(2)
