"""Per-KB vector store backed by Chroma. Replaces the FAISS-file approach."""

import hashlib
import os
import threading
from pathlib import Path
from typing import Any, Callable, List, Optional

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from src.knowledebase.embedding import EmbeddingPipeline

ProgressCallback = Callable[[dict], None]

DATA_DIR = Path(os.getenv("DATA_DIR", ".kb_data"))
CHROMA_DIR = DATA_DIR / "chroma"

_client: Optional[chromadb.ClientAPI] = None
_client_lock = threading.Lock()


def get_chroma_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                CHROMA_DIR.mkdir(parents=True, exist_ok=True)
                _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
                print(f"[INFO] Chroma persistent client initialized at {CHROMA_DIR}")
    return _client


def _chunk_metadata_and_text(chunk: Any) -> tuple[dict, str]:
    """Extract (metadata_without_text, text) from a LangChain chunk. Preserves URL sources verbatim."""
    src = ""
    page: Optional[int] = None
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
    meta: dict = {"source": src}
    if page is not None:
        meta["page"] = page
    return meta, chunk.page_content


def _chunk_id(source: str, page: Optional[int], text: str) -> str:
    payload = f"{source}||{page if page is not None else ''}||{text}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:20]


class ChromaKbStore:
    """Per-KB store: one Chroma collection identified by kb_id."""

    def __init__(
        self,
        kb_id: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model: Optional[SentenceTransformer] = None,
    ):
        self.kb_id = kb_id
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = model if model is not None else SentenceTransformer(embedding_model)
        client = get_chroma_client()
        self.collection = client.get_or_create_collection(name=f"kb_{kb_id}")

    # ---- reads ----

    def is_empty(self) -> bool:
        return self.collection.count() == 0

    def query(self, query_text: str, top_k: int = 5) -> List[dict]:
        if self.is_empty():
            return []
        emb = self.model.encode([query_text], convert_to_numpy=True).astype("float32")
        res = self.collection.query(
            query_embeddings=emb.tolist(),
            n_results=top_k,
            include=["metadatas", "distances", "documents"],
        )
        results: List[dict] = []
        ids = res.get("ids") or [[]]
        metas = res.get("metadatas") or [[]]
        dists = res.get("distances") or [[]]
        docs = res.get("documents") or [[]]
        if not ids or not ids[0]:
            return results
        for i, cid in enumerate(ids[0]):
            meta = dict(metas[0][i] or {}) if metas and metas[0] else {}
            if docs and docs[0]:
                meta["text"] = docs[0][i]
            distance = dists[0][i] if dists and dists[0] else None
            results.append({"index": cid, "metadata": meta, "distance": distance})
        return results

    # ---- writes ----

    def build_from_documents(
        self,
        documents: List[Any],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        print(f"[INFO] Adding {len(documents)} raw documents to KB '{self.kb_id}'...")
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
        if not chunks:
            return

        embeddings = emb_pipe.embed_chunks(chunks, progress_callback=progress_callback)

        if progress_callback:
            progress_callback({"type": "indexing"})

        ids: List[str] = []
        metadatas: List[dict] = []
        texts: List[str] = []
        seen: set[str] = set()
        for chunk, emb in zip(chunks, embeddings):
            meta, text = _chunk_metadata_and_text(chunk)
            cid = _chunk_id(meta.get("source", ""), meta.get("page"), text)
            if cid in seen:
                continue
            seen.add(cid)
            ids.append(cid)
            metadatas.append(meta)
            texts.append(text)

        keep_indices = [i for i, cid in enumerate(ids)]
        embeddings_arr = np.asarray(embeddings, dtype="float32")[: len(chunks)]
        embeddings_kept = embeddings_arr[keep_indices].tolist() if keep_indices else []

        if ids:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings_kept,
                metadatas=metadatas,
                documents=texts,
            )
            print(f"[INFO] Upserted {len(ids)} chunks into KB '{self.kb_id}'")

    def remove_document(self, source_name: str) -> int:
        if self.is_empty():
            return 0
        matching = self.collection.get(where={"source": source_name}, include=[])
        ids = matching.get("ids", []) if matching else []
        if not ids:
            return 0
        self.collection.delete(ids=ids)
        print(f"[INFO] Removed {len(ids)} chunks for source '{source_name}' from KB '{self.kb_id}'")
        return len(ids)

    def delete_collection(self) -> None:
        client = get_chroma_client()
        try:
            client.delete_collection(name=f"kb_{self.kb_id}")
        except Exception as e:
            print(f"[WARN] Chroma delete_collection failed for {self.kb_id}: {e}")
