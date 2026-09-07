export type Citation = { source: string; page?: number };
export type ChatMessage = { role: "user" | "assistant"; content: string; citations?: Citation[] };

export type KbSummary = {
  kb_id: string;
  name: string;
  files_count: number;
  created_at: number;
};

export type SessionInfo = {
  session_id: string;
  active_kb_id: string | null;
  kbs: KbSummary[];
};

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText} - ${body}`);
  }
  return (await res.json()) as T;
}

// ---------- session ----------

export async function createSession(): Promise<SessionInfo> {
  return jsonFetch<SessionInfo>("/api/sessions", { method: "POST" });
}

export async function getSession(sessionId: string): Promise<SessionInfo> {
  return jsonFetch<SessionInfo>(`/api/sessions/${sessionId}`);
}

export async function deleteSession(sessionId: string): Promise<void> {
  await fetch(`/api/sessions/${sessionId}`, { method: "DELETE" });
}

// ---------- KB CRUD ----------

export async function listKbs(sessionId: string): Promise<{ active_kb_id: string | null; kbs: KbSummary[] }> {
  return jsonFetch(`/api/sessions/${sessionId}/kbs`);
}

export async function createKb(sessionId: string, name?: string): Promise<KbSummary> {
  return jsonFetch(`/api/sessions/${sessionId}/kbs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export async function renameKb(sessionId: string, kbId: string, name: string): Promise<{ kb_id: string; name: string }> {
  return jsonFetch(`/api/sessions/${sessionId}/kbs/${kbId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export async function deleteKb(sessionId: string, kbId: string): Promise<{ deleted: boolean; active_kb_id: string | null }> {
  return jsonFetch(`/api/sessions/${sessionId}/kbs/${kbId}`, { method: "DELETE" });
}

export async function setActiveKb(sessionId: string, kbId: string): Promise<{ active_kb_id: string }> {
  return jsonFetch(`/api/sessions/${sessionId}/kbs/${kbId}/active`, { method: "POST" });
}

// ---------- KB documents ----------

export async function listDocuments(sessionId: string, kbId: string): Promise<{ files: string[] }> {
  return jsonFetch(`/api/sessions/${sessionId}/kbs/${kbId}/documents`);
}

export async function deleteDocument(
  sessionId: string,
  kbId: string,
  filename: string
): Promise<{ deleted: string; chunks_removed: number; files: string[] }> {
  return jsonFetch(
    `/api/sessions/${sessionId}/kbs/${kbId}/documents?name=${encodeURIComponent(filename)}`,
    { method: "DELETE" }
  );
}

// ---------- KB history ----------

export async function getHistory(sessionId: string, kbId: string): Promise<{ history: ChatMessage[] }> {
  return jsonFetch(`/api/sessions/${sessionId}/kbs/${kbId}/history`);
}

// ---------- Ingest (SSE) ----------

export type UploadRejection = { name: string; reason: string };

export type UploadProgressEvent =
  | { type: "loading"; file: string; index: number; total: number }
  | { type: "loaded"; file: string; docs: number; index: number; total: number }
  | { type: "chunking" }
  | { type: "chunked"; count: number }
  | { type: "embedding"; done: number; total: number }
  | { type: "indexing" }
  | {
      type: "done";
      accepted: string[];
      rejected: UploadRejection[];
      documents_indexed: number;
      files: string[];
    }
  | { type: "cancelled" }
  | { type: "error"; content: string };

export async function uploadDocumentsStream(
  sessionId: string,
  kbId: string,
  files: File[],
  onEvent: (evt: UploadProgressEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f, f.name));
  await streamSseEvents(
    `/api/sessions/${sessionId}/kbs/${kbId}/documents`,
    { method: "POST", body: form, signal },
    onEvent
  );
}

export async function ingestUrlsStream(
  sessionId: string,
  kbId: string,
  urls: string[],
  onEvent: (evt: UploadProgressEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  await streamSseEvents(
    `/api/sessions/${sessionId}/kbs/${kbId}/urls`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ urls }),
      signal,
    },
    onEvent
  );
}

async function streamSseEvents(
  url: string,
  init: RequestInit,
  onEvent: (evt: UploadProgressEvent) => void
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch (e) {
    if ((e as DOMException)?.name === "AbortError") {
      onEvent({ type: "cancelled" });
      return;
    }
    throw e;
  }
  if (!res.ok || !res.body) {
    const body = await res.text().catch(() => "");
    onEvent({ type: "error", content: `HTTP ${res.status} ${body}` });
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let sep;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        for (const line of frame.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (!payload) continue;
          try {
            onEvent(JSON.parse(payload) as UploadProgressEvent);
          } catch {
            // ignore malformed frame
          }
        }
      }
    }
  } catch (e) {
    if ((e as DOMException)?.name === "AbortError") {
      onEvent({ type: "cancelled" });
      return;
    }
    throw e;
  }
}

// ---------- KB chat ----------

export type ChatStreamCallbacks = {
  onCitations?: (c: Citation[]) => void;
  onToken: (t: string) => void;
  onDone: () => void;
  onError: (msg: string) => void;
};

export async function streamChat(
  sessionId: string,
  kbId: string,
  message: string,
  topK: number,
  cb: ChatStreamCallbacks
): Promise<void> {
  const res = await fetch(`/api/sessions/${sessionId}/kbs/${kbId}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, top_k: topK }),
  });
  if (!res.ok || !res.body) {
    cb.onError(`HTTP ${res.status}`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (!payload) continue;
        try {
          const evt = JSON.parse(payload) as { type: string; content?: unknown };
          if (evt.type === "citations" && Array.isArray(evt.content)) {
            cb.onCitations?.(evt.content as Citation[]);
          } else if (evt.type === "token" && typeof evt.content === "string") {
            cb.onToken(evt.content);
          } else if (evt.type === "done") {
            cb.onDone();
          } else if (evt.type === "error") {
            cb.onError(typeof evt.content === "string" ? evt.content : "unknown");
          }
        } catch {
          // ignore malformed frame
        }
      }
    }
  }
}
