import os 
import faiss
import numpy as np
import pickle
from typing import Callable, List, Any, Optional
from sentence_transformers import SentenceTransformer
from src.knowledebase.embedding import EmbeddingPipeline

ProgressCallback = Callable[[dict], None]


def _chunk_metadata(chunk: Any) -> dict:
    """Build a chunk metadata dict, propagating source file/URL and page from the LangChain document."""
    src = ""
    page = None
    src_meta = getattr(chunk, "metadata", None) or {}
    raw_source = src_meta.get("source")
    if raw_source:
        if isinstance(raw_source, str) and raw_source.startswith(("http://", "https://")):
            src = raw_source
        else:
            src = os.path.basename(raw_source)
    raw_page = src_meta.get("page")
    if isinstance(raw_page, int):
        page = raw_page + 1
    meta: dict = {"text": chunk.page_content, "source": src}
    if page is not None:
        meta["page"] = page
    return meta


class FaissVectorStore:
    def __init__(
        self,
        persist_directory: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model: Optional[SentenceTransformer] = None,
    ):
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        if model is not None:
            self.model = model
        else:
            self.model = SentenceTransformer(embedding_model)
            print(f"[INFO] Loaded embedding model: {embedding_model}")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def build_from_documents(self, documents: List[Any], progress_callback: Optional[ProgressCallback] = None):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            model=self.model,
        )
        if progress_callback:
            progress_callback({"type": "chunking"})
        chunks = emb_pipe.chunk_documents(documents)
        if progress_callback:
            progress_callback({"type": "chunked", "count": len(chunks)})
        embeddings = emb_pipe.embed_chunks(chunks, progress_callback=progress_callback)
        metadata = [_chunk_metadata(chunk) for chunk in chunks]
        if progress_callback:
            progress_callback({"type": "indexing"})
        self.add_embeddings(np.array(embeddings).astype('float32'), metadata)
        self.save()
        print(f"[INFO] Vector store built successfully and saved to {self.persist_directory}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} embeddings to the vector store.")

    def save(self):
        faiss_path = os.path.join(self.persist_directory, "faiss_index.index")
        meta_path = os.path.join(self.persist_directory, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Faiss index and metadata saved to {self.persist_directory}")

    def load(self) -> bool:
        """Load index+metadata from disk if present. Returns False for a fresh/empty store."""
        faiss_path = os.path.join(self.persist_directory, "faiss_index.index")
        meta_path = os.path.join(self.persist_directory, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            print(f"[INFO] No existing Faiss index at {self.persist_directory}, starting empty")
            self.index = None
            self.metadata = []
            return False
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Faiss index and metadata loaded from {self.persist_directory}")
        return True

    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "metadata": meta, "distance": dist})
        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store with text: {query_text}")
        if self.is_empty():
            print("[INFO] Vector store is empty, returning no results")
            return []
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)

    def remove_document(self, source_filename: str) -> int:
        """Remove all chunks whose metadata.source matches source_filename. Rebuilds the FAISS index."""
        if self.is_empty():
            return 0
        keep_indices = [i for i, m in enumerate(self.metadata) if (m or {}).get("source") != source_filename]
        removed = len(self.metadata) - len(keep_indices)
        if removed == 0:
            return 0
        if not keep_indices:
            self.index = None
            self.metadata = []
            self.save()
            return removed
        dim = self.index.d
        kept_vectors = np.vstack([self.index.reconstruct(i) for i in keep_indices]).astype("float32")
        kept_meta = [self.metadata[i] for i in keep_indices]
        new_index = faiss.IndexFlatL2(dim)
        new_index.add(kept_vectors)
        self.index = new_index
        self.metadata = kept_meta
        self.save()
        print(f"[INFO] Removed {removed} chunks for source '{source_filename}'")
        return removed



