import chromadb, uuid
from chromadb.config import Settings
from .embedder import get_embedding
from app.config import TOP_K, CHUNK_SIZE, CHUNK_OVERLAP

client = chromadb.Client(
    Settings(persist_directory="./chroma_db")
)

collection = client.get_or_create_collection(name="documents")


def add_documents(pages: list[tuple[int, str]], source: str):
    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for page_num, page_text in pages:
        chunks = split_text(page_text)
        for chunk in chunks:
            doc_id = f"{source}-{page_num}-{uuid.uuid4().hex[:8]}"
            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append({
                "source": source,
                "page": page_num
            })
            embeddings.append(get_embedding(chunk))

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )


def search(query: str, k: int = TOP_K):
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]  # 작을수록 유사

    return docs, metas, dists


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(text), step):
        chunks.append(text[i:i + chunk_size])
    return chunks