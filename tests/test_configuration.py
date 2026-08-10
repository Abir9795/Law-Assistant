from urllib.error import URLError

import pytest

import app


def test_missing_ollama_has_actionable_error(monkeypatch):
    monkeypatch.setattr(app, "urlopen", lambda *_args, **_kwargs: _raise_url_error())

    with pytest.raises(app.AssistantUnavailableError, match="ollama serve"):
        app._check_ollama()


def _raise_url_error():
    raise URLError("connection refused")
