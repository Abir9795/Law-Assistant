from types import SimpleNamespace

import pytest

import app as app_module


class FakeChain:
    def __init__(self, answer="Test legal response"):
        self.answer = answer
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        return {"answer": self.answer}


@pytest.fixture()
def client(monkeypatch):
    base_chain = FakeChain()
    memory_chain = FakeChain()
    monkeypatch.setattr(
        app_module, "_build_rag_chains", lambda: (base_chain, memory_chain)
    )
    flask_app = app_module.create_app(
        {"TESTING": True, "SECRET_KEY": "test-secret"}
    )
    return flask_app.test_client(), base_chain, memory_chain


def test_homepage_loads(client):
    test_client, _, _ = client
    response = test_client.get("/")
    assert response.status_code == 200
    assert b"Law Assistant" in response.data
    assert b"DOMPurify.sanitize" in response.data


def test_health_endpoint(client):
    test_client, _, _ = client
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_empty_question_is_rejected(client):
    test_client, _, _ = client
    response = test_client.post("/memory-chat", data={"msg": "  "})
    assert response.status_code == 400


def test_memory_chat_calls_chain_and_stores_turns(client):
    test_client, _, memory_chain = client
    response = test_client.post("/memory-chat", data={"msg": "What is bail?"})
    assert response.status_code == 200
    assert response.get_data(as_text=True) == "Test legal response"
    assert memory_chain.calls[0]["input"] == "What is bail?"

    with test_client.session_transaction() as session:
        assert session["history"][-1] == {
            "role": "assistant",
            "content": "Test legal response",
        }


def test_reset_history(client):
    test_client, _, _ = client
    test_client.post("/memory-chat", data={"msg": "What is bail?"})
    response = test_client.post("/reset-history")
    assert response.status_code == 200
    with test_client.session_transaction() as session:
        assert "history" not in session


def test_retrieved_sources_are_appended(monkeypatch):
    class SourceChain:
        def invoke(self, _payload):
            return {
                "answer": "Supported answer",
                "context": [
                    SimpleNamespace(
                        metadata={"source": "Data/evidence-act.pdf", "page": 2}
                    )
                ],
            }

    chain = SourceChain()
    monkeypatch.setattr(app_module, "_build_rag_chains", lambda: (chain, chain))
    flask_app = app_module.create_app(
        {"TESTING": True, "SECRET_KEY": "test-secret"}
    )
    response = flask_app.test_client().post("/get", data={"msg": "Evidence?"})
    assert response.status_code == 200
    assert "evidence-act.pdf, page 3" in response.get_data(as_text=True)
