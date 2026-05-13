import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

def ingest_pdf():
    print("="*65)
    print("\t\t\tStarting Ingestion")
    print("="*65)
    print(f"PDF: {PDF_PATH}\nCHUNK_SIZE: {CHUNK_SIZE}\nCHUNK_OVERLAP: {CHUNK_OVERLAP}")

    project_root = Path(__file__).parent.parent
    loader = PyPDFLoader(str(project_root/PDF_PATH))

    docs = loader.load()

    splits = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_documents(docs)
    if not splits:
        print("No data to split")
        raise SystemExit(0)

    print("Successfully splitted docs")
    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
        )
        for d in splits
    ]
    ids = [f"part-{i}" for i in range(len(enriched))]

    print("Enriches done")

    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL","text-embedding-3-small"))

    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )

    print("Uploading to PGVector...")

    store.add_documents(documents=enriched, ids=ids)
    print("="*65)
    print("Ingestion made successfully!")
    pass


if __name__ == "__main__":
    ingest_pdf()