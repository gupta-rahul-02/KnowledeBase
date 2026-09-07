import { useRef, useState } from "react";
import { deleteDocument, ingestUrlsStream, uploadDocumentsStream, UploadProgressEvent } from "../api/client";

type Props = {
  sessionId: string;
  kbId: string;
  files: string[];
  onFilesChanged: (files: string[]) => void;
};

type Progress = { label: string; done?: number; total?: number };

function eventToProgress(evt: UploadProgressEvent): Progress | null {
  switch (evt.type) {
    case "loading":
      return { label: `Fetching ${evt.file} (${evt.index}/${evt.total})` };
    case "loaded":
      return { label: `Fetched ${evt.file} — ${evt.docs} page(s)` };
    case "chunking":
      return { label: "Chunking documents..." };
    case "chunked":
      return { label: `Chunked into ${evt.count} piece(s)` };
    case "embedding": {
      const pct = evt.total > 0 ? Math.round((evt.done / evt.total) * 100) : 0;
      return { label: `Embedding chunks (${pct}%)`, done: evt.done, total: evt.total };
    }
    case "indexing":
      return { label: "Building index..." };
    default:
      return null;
  }
}

export function FileUploadPanel({ sessionId, kbId, files, onFilesChanged }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [urlInput, setUrlInput] = useState("");

  async function runIngest(
    initialLabel: string,
    ingest: (onEvent: (evt: UploadProgressEvent) => void, signal: AbortSignal) => Promise<void>
  ) {
    const controller = new AbortController();
    abortRef.current = controller;
    setBusy(true);
    setError(null);
    setStatus(null);
    setProgress({ label: initialLabel });
    try {
      await ingest((evt) => {
        if (evt.type === "done") {
          onFilesChanged(evt.files);
          const rejected = evt.rejected.length
            ? ` (skipped ${evt.rejected.length}: ${evt.rejected.map((r) => r.name).join(", ")})`
            : "";
          setStatus(`Indexed ${evt.accepted.length} item(s), ${evt.documents_indexed} document(s)${rejected}`);
          setProgress(null);
        } else if (evt.type === "cancelled") {
          setStatus("Cancelled");
          setProgress(null);
        } else if (evt.type === "error") {
          setError(evt.content);
          setProgress(null);
        } else {
          const p = eventToProgress(evt);
          if (p) setProgress(p);
        }
      }, controller.signal);
    } catch (e) {
      setError(String(e));
      setProgress(null);
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  function handleStop() {
    abortRef.current?.abort();
  }

  async function handleFiles(selected: FileList | null) {
    if (!selected || selected.length === 0) return;
    const arr = Array.from(selected);
    await runIngest(
      `Uploading ${arr.length} file(s)...`,
      (onEvent, signal) => uploadDocumentsStream(sessionId, kbId, arr, onEvent, signal)
    );
    if (inputRef.current) inputRef.current.value = "";
  }

  async function handleAddUrl() {
    const url = urlInput.trim();
    if (!url || busy) return;
    await runIngest(
      `Fetching ${url}...`,
      (onEvent, signal) => ingestUrlsStream(sessionId, kbId, [url], onEvent, signal)
    );
    setUrlInput("");
  }

  async function handleDelete(filename: string) {
    if (busy) return;
    if (!confirm(`Remove "${filename}" from the knowledge base?`)) return;
    setBusy(true);
    setError(null);
    setStatus(`Removing ${filename}...`);
    try {
      const result = await deleteDocument(sessionId, kbId, filename);
      onFilesChanged(result.files);
      setStatus(`Removed ${filename} (${result.chunks_removed} chunks)`);
    } catch (e) {
      setError(String(e));
      setStatus(null);
    } finally {
      setBusy(false);
    }
  }

  const percent =
    progress && progress.total && progress.total > 0
      ? Math.round(((progress.done ?? 0) / progress.total) * 100)
      : null;

  return (
    <div className="upload-panel">
      <h3>Knowledge Base</h3>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".pdf,.txt,.csv,.docx,.xlsx,.json,.md"
        disabled={busy}
        onChange={(e) => handleFiles(e.target.files)}
      />
      <div className="url-row">
        <input
          type="url"
          className="url-input"
          placeholder="Or paste a webpage URL..."
          value={urlInput}
          disabled={busy}
          onChange={(e) => setUrlInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              handleAddUrl();
            }
          }}
        />
        <button
          type="button"
          className="url-add"
          disabled={busy || !urlInput.trim()}
          onClick={handleAddUrl}
        >
          Add
        </button>
      </div>
      {progress && (
        <div className="progress">
          <div className="progress-label">
            <span>{progress.label}</span>
            {progress.total ? (
              <span className="progress-count">
                {progress.done ?? 0}/{progress.total}
              </span>
            ) : null}
          </div>
          {percent !== null ? (
            <div className="progress-bar">
              <div className="progress-bar-fill" style={{ width: `${percent}%` }} />
            </div>
          ) : (
            <div className="progress-bar progress-bar-indeterminate">
              <div className="progress-bar-fill" />
            </div>
          )}
          <button type="button" className="stop-btn" onClick={handleStop}>
            Stop
          </button>
        </div>
      )}
      {status && <div className="status">{status}</div>}
      {error && <div className="error">{error}</div>}
      <div className="file-list">
        <div className="file-list-title">Documents ({files.length})</div>
        {files.length === 0 ? (
          <div className="muted">No documents yet. Upload some to start.</div>
        ) : (
          <ul>
            {files.map((f) => {
              const isUrl = f.startsWith("http://") || f.startsWith("https://");
              return (
                <li key={f} className="file-item">
                  <span className={`file-badge ${isUrl ? "file-badge-url" : "file-badge-file"}`}>
                    {isUrl ? "URL" : "FILE"}
                  </span>
                  <span className="file-name" title={f}>{f}</span>
                  <button
                    type="button"
                    className="file-remove"
                    disabled={busy}
                    onClick={() => handleDelete(f)}
                    title={`Remove ${f}`}
                  >
                    ×
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}
