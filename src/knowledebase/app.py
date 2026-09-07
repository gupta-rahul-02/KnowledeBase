from src.knowledebase.data_loader import load_all_documents
from src.knowledebase.vector_store import FaissVectorStore
from src.knowledebase.search import RAGSearch


## Example Usage

if __name__ == "__main__":
    # docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")

    # store.build_from_documents(docs)
    store.load()
    print(store.query("What is referral policy?", top_k=3))
    
    # Example query
    # query_text = "example query"
    # results = store.query(query_text, top_k=5)
    # print(f"[INFO] Query results for '{query_text}': {results}")  

    #RAG Search
    rag_search = RAGSearch()
    query = "What is referral policy?"
    summary = rag_search.search_and_Summarize(query, top_k=3)
    print(f"[INFO] Summary for query '{query}': {summary}")

