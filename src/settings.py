import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _clean(value: str | None, default: str = "") -> str:
    if value is None:
        return default
    return value.strip().strip("'\"").strip()


def database_url() -> str:
    raw = _clean(os.getenv("DATABASE_URL"))
    if not raw:
        raise RuntimeError(
            "DATABASE_URL não definida. Copie .env.example para .env e preencha a conexão."
        )
    if raw.startswith("postgres://"):
        raw = "postgresql+psycopg://" + raw[len("postgres://") :]
    elif raw.startswith("postgresql://") and not raw.startswith("postgresql+"):
        raw = "postgresql+psycopg://" + raw[len("postgresql://") :]
    return raw


def collection_name() -> str:
    return _clean(os.getenv("PG_VECTOR_COLLECTION_NAME"), "documents")


def pdf_path() -> Path:
    raw = _clean(os.getenv("PDF_PATH"), "document.pdf")
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def provider() -> str:
    forced = _clean(os.getenv("LLM_PROVIDER")).lower()
    if forced in {"openai", "gemini", "google"}:
        selected = "gemini" if forced == "google" else forced
    elif _clean(os.getenv("OPENAI_API_KEY")):
        selected = "openai"
    elif _clean(os.getenv("GOOGLE_API_KEY")):
        selected = "gemini"
    else:
        raise RuntimeError(
            "Defina OPENAI_API_KEY ou GOOGLE_API_KEY no arquivo .env. "
            "Use LLM_PROVIDER=openai ou LLM_PROVIDER=gemini se as duas chaves existirem."
        )

    if selected == "openai" and not _clean(os.getenv("OPENAI_API_KEY")):
        raise RuntimeError("OPENAI_API_KEY não definida para LLM_PROVIDER=openai.")
    if selected == "gemini" and not _clean(os.getenv("GOOGLE_API_KEY")):
        raise RuntimeError("GOOGLE_API_KEY não definida para LLM_PROVIDER=gemini.")
    return selected


def build_embeddings():
    selected = provider()
    if selected == "openai":
        from langchain_openai import OpenAIEmbeddings

        model = _clean(os.getenv("OPENAI_EMBEDDING_MODEL"), "text-embedding-3-small")
        return OpenAIEmbeddings(
            model=model,
            api_key=_clean(os.getenv("OPENAI_API_KEY")),
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    model = _clean(os.getenv("GOOGLE_EMBEDDING_MODEL"), "models/gemini-embedding-001")
    return GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=_clean(os.getenv("GOOGLE_API_KEY")),
    )


def build_llm():
    selected = provider()
    if selected == "openai":
        from langchain_openai import ChatOpenAI

        model = _clean(os.getenv("OPENAI_LLM_MODEL"), "gpt-4o-mini")
        return ChatOpenAI(
            model=model,
            temperature=0,
            api_key=_clean(os.getenv("OPENAI_API_KEY")),
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    model = _clean(os.getenv("GOOGLE_LLM_MODEL"), "gemini-flash-lite-latest")
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        google_api_key=_clean(os.getenv("GOOGLE_API_KEY")),
    )


def build_vectorstore(embeddings=None, *, pre_delete_collection: bool = False):
    from langchain_postgres import PGVector

    return PGVector(
        embeddings=embeddings or build_embeddings(),
        collection_name=collection_name(),
        connection=database_url(),
        use_jsonb=True,
        pre_delete_collection=pre_delete_collection,
    )
