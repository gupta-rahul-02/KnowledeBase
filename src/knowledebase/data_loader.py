from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredExcelLoader, UnstructuredWordDocumentLoader
from langchain_community.document_loaders import JSONLoader, UnstructuredMarkdownLoader
from langchain_community.document_loaders import WebBaseLoader

# Maps file extension -> loader class used for that file type
_LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".docx": Docx2txtLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".json": JSONLoader,
    ".md": UnstructuredMarkdownLoader,
}

SUPPORTED_EXTENSIONS = tuple(_LOADER_MAP.keys())


def _load_single_file(file_path: Path) -> List[Any]:
    """Load a single file into LangChain Document(s) based on its extension."""
    ext = file_path.suffix.lower()
    loader_cls = _LOADER_MAP.get(ext)
    if loader_cls is None:
        print(f"[WARN] Unsupported file type '{ext}' for {file_path}, skipping")
        return []
    try:
        loader = loader_cls(str(file_path))
        loaded = loader.load()
        print(f"[DEBUG] Loaded {len(loaded)} docs from {file_path}")
        return loaded
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return []


def load_documents_from_paths(file_paths: List[str]) -> List[Any]:
    """Load documents from an explicit list of file paths (used for uploads)."""
    documents: List[Any] = []
    for p in file_paths:
        documents.extend(_load_single_file(Path(p)))
    return documents


def is_valid_url(url: str) -> bool:
    """Basic sanity check: URL must be http(s) with a host."""
    if not isinstance(url, str):
        return False
    url = url.strip()
    return url.startswith(("http://", "https://")) and len(url) > len("https://")


def _load_single_url(url: str, timeout: int = 15) -> List[Any]:
    """Fetch a single URL and return LangChain Document(s). Sets a UA + timeout to avoid hangs."""
    try:
        loader = WebBaseLoader(
            web_paths=[url],
            requests_kwargs={
                "headers": {"User-Agent": "KnowledeBaseRAG/0.1 (+https://local)"},
                "timeout": timeout,
            },
        )
        loaded = loader.load()
        print(f"[DEBUG] Loaded {len(loaded)} docs from {url}")
        return loaded
    except Exception as e:
        print(f"[ERROR] Failed to load URL {url}: {e}")
        return []


def load_documents_from_urls(urls: List[str], timeout: int = 15) -> List[Any]:
    """Load documents from an explicit list of URLs.

    For a single URL, does a straightforward blocking fetch.
    For multiple URLs, uses WebBaseLoader's batched fetching so I/O happens concurrently.
    """
    if not urls:
        return []
    if len(urls) == 1:
        return _load_single_url(urls[0], timeout=timeout)
    try:
        loader = WebBaseLoader(
            web_paths=urls,
            requests_kwargs={
                "headers": {"User-Agent": "KnowledeBaseRAG/0.1 (+https://local)"},
                "timeout": timeout,
            },
        )
        loader.requests_per_second = 5
        loaded = loader.load()
        print(f"[DEBUG] Loaded {len(loaded)} docs from {len(urls)} URLs (batched)")
        return loaded
    except Exception as e:
        print(f"[ERROR] Batched URL load failed: {e}; falling back to sequential")
        docs: List[Any] = []
        for u in urls:
            docs.extend(_load_single_url(u, timeout=timeout))
        return docs


def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the data directory and convert to Langchain document structure.
    Supported file types: .pdf, .txt, .csv, .docx, .xlsx, .json, .md
    """
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents: List[Any] = []
    for ext in SUPPORTED_EXTENSIONS:
        files = list(data_path.glob(f"**/*{ext}"))
        print(f"[DEBUG] Found {len(files)} {ext} files")
        for f in files:
            documents.extend(_load_single_file(f))
    return documents
