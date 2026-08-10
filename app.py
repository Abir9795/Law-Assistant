"""Flask application for the Bangladesh Law Assistant."""

import logging
import os
import secrets
import sys
from functools import lru_cache
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from dotenv import load_dotenv
from flask import Flask, render_template, request, session

from src.prompt import memory_prompt, system_prompt

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    stream=sys.stdout,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "2000"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))


class AssistantUnavailableError(RuntimeError):
    """A safe configuration/runtime error that can be displayed to the user."""


def _required_setting(name: str) -> str:
    """Return a required environment setting with a helpful error."""
    value = os.getenv(name, "").strip()
    if not value:
        raise AssistantUnavailableError(
            f"{name} is not configured. Copy .env.example to .env and add a value."
        )
    return value


def _check_ollama() -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    try:
        with urlopen(f"{base_url}/api/tags", timeout=3) as response:
            if response.status != 200:
                raise URLError(f"unexpected status {response.status}")
    except (OSError, URLError) as error:
        raise AssistantUnavailableError(
            "Ollama is not running. Start it with `ollama serve`, then try again."
        ) from error
    return base_url


@lru_cache(maxsize=1)
def _build_rag_chains():
    """Build expensive embedding, retrieval, and LLM objects on first use."""
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_community.llms import Ollama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
    from langchain_pinecone import PineconeVectorStore

    from src.helper import download_hugging_face_embeddings
    from src.legal_lookup import lookup_exact_sections

    ollama_base_url = _check_ollama()
    os.environ["PINECONE_API_KEY"] = _required_setting("PINECONE_API_KEY")
    embedding_model = download_hugging_face_embeddings()
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=os.getenv("PINECONE_INDEX", "lawbot"),
        embedding=embedding_model,
    )
    semantic_retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": int(os.getenv("RETRIEVAL_TOP_K", "3"))},
    )

    def hybrid_retrieve(payload):
        query = payload["input"] if isinstance(payload, dict) else str(payload)
        documents = lookup_exact_sections(query)
        documents.extend(semantic_retriever.invoke(query))
        unique_documents = []
        seen_content = set()
        for document in documents:
            content_key = document.page_content.strip()
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_documents.append(document)
        return unique_documents

    retriever = RunnableLambda(hybrid_retrieve)
    llm = Ollama(
        model=os.getenv("OLLAMA_MODEL", "llama3"),
        temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.4")),
        base_url=ollama_base_url,
    )

    base_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )
    base_chain = create_retrieval_chain(
        retriever, create_stuff_documents_chain(llm, base_prompt)
    )

    memory_template = ChatPromptTemplate.from_messages(
        [
            ("system", memory_prompt),
            ("system", "Chat history so far:\n{history}"),
            ("human", "{input}"),
        ]
    )
    memory_chain = create_retrieval_chain(
        retriever, create_stuff_documents_chain(llm, memory_template)
    )
    return base_chain, memory_chain


def _get_history() -> list[dict[str, str]]:
    if "history" not in session:
        session["history"] = []
    return session["history"]


def _history_text() -> str:
    return "\n".join(
        f"{message['role']}: {message['content']}" for message in _get_history()
    )


def _save_turn(role: str, content: str) -> None:
    history = _get_history()
    history.append({"role": role, "content": content})
    session["history"] = history[-MAX_HISTORY_MESSAGES:]
    session.modified = True


def _get_message() -> str:
    message = (request.form.get("msg") or "").strip()
    if not message:
        message = str((request.get_json(silent=True) or {}).get("msg", "")).strip()
    return message


def _format_sources(documents: list) -> str:
    """Create a small, deduplicated Markdown source list from retrieved documents."""
    sources = []
    seen = set()
    for document in documents or []:
        metadata = getattr(document, "metadata", {})
        raw_source = metadata.get("source") or metadata.get("file_path")
        if not raw_source:
            continue
        label = Path(str(raw_source)).name
        page = metadata.get("page")
        if isinstance(page, int):
            label = f"{label}, page {page + 1}"
        if label not in seen:
            seen.add(label)
            sources.append(label)
        if len(sources) == 3:
            break
    if not sources:
        return ""
    return "\n\n**Retrieved sources**\n" + "\n".join(
        f"- {source}" for source in sources
    )


def _run_chat(use_memory: bool):
    message = _get_message()
    if not message:
        return "Please enter a question.", 400
    if len(message) > MAX_MESSAGE_LENGTH:
        return f"Question must be at most {MAX_MESSAGE_LENGTH} characters.", 400

    if os.getenv("LOG_CHAT_CONTENT", "false").lower() == "true":
        logger.info("User question: %s", message)

    try:
        base_chain, memory_chain = _build_rag_chains()
        if use_memory:
            _save_turn("user", message)
            response = memory_chain.invoke(
                {"input": message, "history": _history_text()}
            )
        else:
            response = base_chain.invoke({"input": message})

        answer = (response.get("answer") or "Sorry, no answer was found.").strip()
        answer += _format_sources(response.get("context", []))
        if use_memory:
            _save_turn("assistant", answer)
        return answer, 200
    except AssistantUnavailableError as error:
        logger.warning("Assistant unavailable: %s", error)
        return str(error), 503
    except Exception:
        logger.exception("Chat request failed")
        return (
            "The assistant is temporarily unavailable. Check the server logs, "
            "Pinecone configuration, and Ollama status.",
            503,
        )


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    configured_secret = os.getenv("SECRET_KEY")
    if not configured_secret:
        logger.warning(
            "SECRET_KEY is unset; using a temporary key for local development."
        )
    app.config.from_mapping(
        SECRET_KEY=configured_secret or secrets.token_hex(32),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        MAX_CONTENT_LENGTH=16 * 1024,
    )
    if test_config:
        app.config.update(test_config)

    @app.get("/")
    def index():
        return render_template("chat.html")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.after_request
    def add_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    @app.post("/get")
    def chat():
        return _run_chat(use_memory=False)

    @app.post("/memory-chat")
    def memory_chat():
        return _run_chat(use_memory=True)

    @app.post("/reset-history")
    def reset_history():
        session.pop("history", None)
        return "History cleared.", 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8080")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
