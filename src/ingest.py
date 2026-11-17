import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import build_embeddings, collection_name, database_url, pdf_path, provider

GEMINI_BATCH_SIZE = 8
GEMINI_PAUSE_SECONDS = 45


def ingest_pdf():
    path = pdf_path()
    if not path.is_file():
        raise FileNotFoundError(f"PDF não encontrado: {path}")

    print(f"Lendo PDF: {path}")
    documents = PyPDFLoader(str(path)).load()
    if not documents:
        raise RuntimeError("O PDF não retornou páginas.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = [
        chunk
        for chunk in splitter.split_documents(documents)
        if chunk.page_content and chunk.page_content.strip()
    ]
    if not chunks:
        raise RuntimeError("Nenhum trecho de texto foi extraído do PDF.")

    selected = provider()
    print(f"Páginas: {len(documents)} | Chunks: {len(chunks)}")
    print(f"Provedor de embeddings: {selected}")
    print("Gerando embeddings e gravando no Postgres...")

    store = PGVector(
        embeddings=build_embeddings(),
        collection_name=collection_name(),
        connection=database_url(),
        use_jsonb=True,
        pre_delete_collection=True,
    )
    batch_size = GEMINI_BATCH_SIZE if selected == "gemini" else len(chunks)
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        end = start + len(batch)
        print(f"Gravando chunks {start + 1}-{end} de {len(chunks)}...")
        _add_documents(store, batch)
        if selected == "gemini" and end < len(chunks):
            print(f"Aguardando {GEMINI_PAUSE_SECONDS}s para respeitar a cota gratuita...")
            time.sleep(GEMINI_PAUSE_SECONDS)
    print(f"Ingestão concluída na collection '{collection_name()}'.")


def _add_documents(store, batch):
    for attempt in range(1, 4):
        try:
            store.add_documents(batch)
            return
        except Exception as exc:
            message = str(exc).lower()
            if "429" not in message and "quota" not in message:
                raise
            if attempt == 3:
                raise
            print(f"Cota de embeddings atingida. Nova tentativa em 60s ({attempt}/3)...")
            time.sleep(60)


if __name__ == "__main__":
    try:
        ingest_pdf()
    except Exception as exc:
        print(f"Falha na ingestão: {exc}")
        raise SystemExit(1) from exc
