from typing import Any, Callable, List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer # pip install sentence-transformers
import numpy as np
from src.knowledebase.data_loader import load_all_documents

ProgressCallback = Callable[[dict], None]


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model: Optional[SentenceTransformer] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if model is not None:
            self.model = model
            print(f"[INFO] Reusing embedding model (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})")
        else:
            self.model = SentenceTransformer(model_name)
            print(f"[INFO] Loaded embedding model: {model_name}, chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}")

    def chunk_documents(self,documents:list[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    def embed_chunks(
        self,
        chunks: list[Any],
        batch_size: int = 32,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        total = len(texts)
        print(f"[INFO] Embedding {total} chunks (batch_size={batch_size})")
        if total == 0:
            return np.zeros((0, 0), dtype=np.float32)
        batches: List[np.ndarray] = []
        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            emb = self.model.encode(batch, convert_to_numpy=True)
            batches.append(emb)
            if progress_callback:
                done = min(i + batch_size, total)
                progress_callback({"type": "embedding", "done": done, "total": total})
        embeddings = np.vstack(batches)
        print(f"[INFO] Generated embeddings with shape: {embeddings.shape}")
        return embeddings




