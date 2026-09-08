"""Per-KB vector store backed by Qdrant Cloud. One collection per KB."""

import os
import uuid
from typing import Any, Callable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from src.knowledebase.embedding import EmbeddingPipeline

ProgressCallback = Callable[[dict], None]

EMBEDDING_DIM = 384

_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        if not url:
            raise RuntimeError("QDRANT_URL is not set")
        _client = QdrantClient(url=url, api_key=api_key, prefer_grpc=False)
        print(f"[INFO] Qdrant client initialized at {url}")
    return _client


def _chunk_metadata_and_text(chunk: Any) -> tuple[dict, str]:
    """Extract (metadata, text) from a LangChain chunk. Preserves URL sources verbatim."""
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
    """Deterministic UUID from (source, page, text) — Qdrant requires UUID or unsigned int IDs."""
    payload = f"{source}||{page if page is not None else ''}||{text}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, payload))


class QdrantKbStore:
    """Per-KB store: one Qdrant collection identified by kb_id."""

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
        self.collection_name = f"kb_{kb_id}"
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        client = get_qdrant_client()
        if not client.collection_exists(self.collection_name):
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            print(f"[INFO] Created Qdrant collection '{self.collection_name}'")
        # Payload index on `source` is required for filter-based deletes (remove_document).
        try:
            client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    # ---- reads ----

    def is_empty(self) -> bool:
        return get_qdrant_client().count(self.collection_name, exact=True).count == 0

    def query(self, query_text: str, top_k: int = 5) -> List[dict]:
        if self.is_empty():
            return []
        emb = self.model.encode([query_text], convert_to_numpy=True).astype("float32")
        client = get_qdrant_client()
        res = client.query_points(
            collection_name=self.collection_name,
            query=emb[0].tolist(),
            limit=top_k,
            with_payload=True,
        )
        results: List[dict] = []
        for point in res.points:
            payload = dict(point.payload or {})
            results.append({"index": str(point.id), "metadata": payload, "distance": point.score})
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

        points: List[PointStruct] = []
        seen: set[str] = set()
        for chunk, emb in zip(chunks, embeddings):
            meta, text = _chunk_metadata_and_text(chunk)
            cid = _chunk_id(meta.get("source", ""), meta.get("page"), text)
            if cid in seen:
                continue
            seen.add(cid)
            payload = dict(meta)
            payload["text"] = text
            points.append(PointStruct(id=cid, vector=emb.tolist(), payload=payload))

        if points:
            get_qdrant_client().upsert(collection_name=self.collection_name, points=points)
            print(f"[INFO] Upserted {len(points)} chunks into KB '{self.kb_id}'")

    def remove_document(self, source_name: str) -> int:
        client = get_qdrant_client()
        selector = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_name))]
        )
        count = client.count(
            collection_name=self.collection_name, count_filter=selector, exact=True
        ).count
        if count > 0:
            client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(filter=selector),
            )
            print(f"[INFO] Removed {count} chunks for source '{source_name}' from KB '{self.kb_id}'")
        return count

    def delete_collection(self) -> None:
        client = get_qdrant_client()
        try:
            client.delete_collection(collection_name=self.collection_name)
        except Exception as e:
            print(f"[WARN] Qdrant delete_collection failed for {self.kb_id}: {e}")
