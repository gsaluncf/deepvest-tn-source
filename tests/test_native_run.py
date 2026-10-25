import socket

import pytest

from deepvest_tn.native_run import governed_server


def test_runner_refuses_an_occupied_port_before_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "deepvest_tn.native_run.subprocess.Popen",
        lambda *_a, **_k: pytest.fail("process launched on an occupied port"),
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with (
            pytest.raises(ValueError, match="already occupied"),
            governed_server(
                binary=tmp_path / "server",
                model=tmp_path / "model",
                config=tmp_path / "config.json",
                port=listener.getsockname()[1],
                api_key="test",
            ),
        ):
            pytest.fail("another service's health response was accepted")
