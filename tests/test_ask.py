from types import SimpleNamespace

from app import ask as ask_module


def test_ask_prints_token_usage_and_returns_response(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=7,
        ),
        text="回答です",
    )

    call = {}

    def generate_content(**kwargs):
        call.update(kwargs)
        return response

    class FakeClient:
        models = SimpleNamespace(generate_content=generate_content)

    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: FakeClient())
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.delenv("AIE_TEMPERATURE", raising=False)

    assert ask_module.ask("質問です") == "回答です"
    output = capsys.readouterr().out
    assert "入力: 12" in output
    assert "出力: 7" in output
    assert "config" not in call


def test_ask_passes_and_prints_temperature_when_set(monkeypatch, capsys):
    response = SimpleNamespace(usage_metadata=None, text="回答です")
    call = {}

    def generate_content(**kwargs):
        call.update(kwargs)
        return response

    class FakeClient:
        models = SimpleNamespace(generate_content=generate_content)

    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: FakeClient())
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setenv("AIE_TEMPERATURE", "0.4")

    assert ask_module.ask("質問です") == "回答です"
    assert call["config"].temperature == 0.4
    assert "[temperature] 0.4" in capsys.readouterr().out