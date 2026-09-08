import os
from typing import Any, Iterator, List, Optional, Tuple
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

ChatMessage = dict  # {"role": "user"|"assistant", "content": str, "citations"?: [...]}
Citation = dict  # {"source": str, "page"?: int}


class RAGSearch:
    def __init__(
        self,
        vectorstore: Any,
        llm_model: str = "openai/gpt-oss-20b",
    ):
        self.vectorstore = vectorstore
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set in environment")
        self.llm = ChatGroq(api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized with model: {llm_model}")

    def _retrieve_context(self, query: str, top_k: int) -> Tuple[str, List[Citation]]:
        results = self.vectorstore.query(query, top_k=top_k)
        texts: List[str] = []
        citations: List[Citation] = []
        seen: set = set()
        for r in results:
            meta = r.get("metadata") or {}
            text = meta.get("text", "")
            if text:
                texts.append(text)
            src = meta.get("source", "")
            if src:
                page = meta.get("page")
                key = (src, page)
                if key not in seen:
                    seen.add(key)
                    cite: Citation = {"source": src}
                    if page is not None:
                        cite["page"] = page
                    citations.append(cite)
        return "\n\n---\n\n".join(texts), citations

    def _build_prompt(self, query: str, context: str, history: Optional[List[ChatMessage]]) -> str:
        history_block = ""
        if history:
            turns = [f"{m['role'].capitalize()}: {m['content']}" for m in history]
            history_block = "Conversation so far:\n" + "\n".join(turns) + "\n\n"
        if not context:
            return (
                f"{history_block}"
                f"You are a helpful assistant. The user asked: '{query}'.\n"
                "No relevant context was found in the knowledge base. "
                "Reply that you don't have information on this topic in the uploaded documents."
            )
        return (
            f"{history_block}"
            "You are a helpful assistant answering questions strictly from the provided context. "
            "If the answer is not in the context, say you don't know.\n\n"
            "Formatting rules:\n"
            "- Answer directly, without preamble like 'Based on the context provided'.\n"
            "- Keep it concise: short paragraphs and/or bullet points.\n"
            "- Use a Markdown table only when comparing multiple items or listing structured data.\n"
            "- Use **bold** only for key terms, not whole sentences.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

    def search_and_Summarize(
        self,
        query: str,
        top_k: int = 5,
        history: Optional[List[ChatMessage]] = None,
    ) -> str:
        context, _ = self._retrieve_context(query, top_k)
        if not context and not history:
            return "No relevant context found."
        prompt = self._build_prompt(query, context, history)
        response = self.llm.invoke(prompt)
        return response.content

    def retrieve_citations(self, query: str, top_k: int = 5) -> List[Citation]:
        _, citations = self._retrieve_context(query, top_k)
        return citations

    def stream_answer(
        self,
        query: str,
        top_k: int = 5,
        history: Optional[List[ChatMessage]] = None,
    ) -> Tuple[List[Citation], Iterator[str]]:
        """Retrieve context and return (citations, token_iterator). Retrieval happens up front."""
        context, citations = self._retrieve_context(query, top_k)
        prompt = self._build_prompt(query, context, history)

        def _iter() -> Iterator[str]:
            for chunk in self.llm.stream(prompt):
                piece = getattr(chunk, "content", "") or ""
                if piece:
                    yield piece

        return citations, _iter()
