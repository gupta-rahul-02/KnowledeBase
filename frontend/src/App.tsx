import { useCallback, useEffect, useState } from "react";
import { ChatWindow } from "./components/ChatWindow";
import { FileUploadPanel } from "./components/FileUploadPanel";
import { KbSwitcher } from "./components/KbSwitcher";
import { useSession } from "./hooks/useSession";
import {
  ChatMessage,
  createKb,
  deleteKb,
  deleteSession,
  getHistory,
  KbSummary,
  listDocuments,
  listKbs,
  renameKb,
  setActiveKb,
} from "./api/client";

export default function App() {
  const { session, setSession, loading, error, persistActiveKb, resetLocalSession } = useSession();
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [files, setFiles] = useState<string[]>([]);
  const [kbBusy, setKbBusy] = useState(false);

  const sessionId = session?.session_id ?? null;
  const activeKbId = session?.active_kb_id ?? null;

  const loadKbContent = useCallback(async () => {
    if (!sessionId || !activeKbId) {
      setHistory([]);
      setFiles([]);
      return;
    }
    try {
      const [h, f] = await Promise.all([
        getHistory(sessionId, activeKbId),
        listDocuments(sessionId, activeKbId),
      ]);
      setHistory(h.history);
      setFiles(f.files);
    } catch (e) {
      console.error("Failed to load KB content:", e);
      setHistory([]);
      setFiles([]);
    }
  }, [sessionId, activeKbId]);

  useEffect(() => {
    loadKbContent();
  }, [loadKbContent]);

  async function refreshKbs() {
    if (!sessionId) return;
    const info = await listKbs(sessionId);
    setSession((prev) => (prev ? { ...prev, active_kb_id: info.active_kb_id, kbs: info.kbs } : prev));
  }

  async function handleSelectKb(kbId: string) {
    if (!sessionId || kbId === activeKbId) return;
    setKbBusy(true);
    try {
      await setActiveKb(sessionId, kbId);
      persistActiveKb(kbId);
      setSession((prev) => (prev ? { ...prev, active_kb_id: kbId } : prev));
    } finally {
      setKbBusy(false);
    }
  }

  async function handleCreateKb(name: string) {
    if (!sessionId) return;
    setKbBusy(true);
    try {
      const kb = await createKb(sessionId, name);
      await setActiveKb(sessionId, kb.kb_id);
      persistActiveKb(kb.kb_id);
      const info = await listKbs(sessionId);
      setSession((prev) =>
        prev ? { ...prev, active_kb_id: kb.kb_id, kbs: info.kbs } : prev
      );
    } finally {
      setKbBusy(false);
    }
  }

  async function handleRenameKb(kbId: string, name: string) {
    if (!sessionId) return;
    setKbBusy(true);
    try {
      await renameKb(sessionId, kbId, name);
      await refreshKbs();
    } finally {
      setKbBusy(false);
    }
  }

  async function handleDeleteKb(kbId: string) {
    if (!sessionId) return;
    setKbBusy(true);
    try {
      const res = await deleteKb(sessionId, kbId);
      if (res.active_kb_id) persistActiveKb(res.active_kb_id);
      await refreshKbs();
    } finally {
      setKbBusy(false);
    }
  }

  // Called from FileUploadPanel when uploads/URL ingests complete or files are deleted
  function handleFilesChanged(newFiles: string[]) {
    setFiles(newFiles);
    setSession((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        kbs: prev.kbs.map((k: KbSummary) =>
          k.kb_id === activeKbId ? { ...k, files_count: newFiles.length } : k
        ),
      };
    });
  }

  async function handleResetSession() {
    if (!session) return;
    if (!confirm("Delete this entire session (all knowledge bases) and start over?")) return;
    try {
      await deleteSession(session.session_id);
    } catch {
      /* ignore */
    }
    resetLocalSession();
    setHistory([]);
    setFiles([]);
    window.location.reload();
  }

  if (loading) return <div className="app-loading">Loading...</div>;
  if (error || !session)
    return <div className="app-error">Failed to start session: {error ?? "unknown"}</div>;

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">Knowledge Base</div>
        <div className="session-id" title={session.session_id}>
          Session: {session.session_id.slice(0, 8)}...
        </div>
        <KbSwitcher
          kbs={session.kbs}
          activeKbId={activeKbId}
          busy={kbBusy}
          onSelect={handleSelectKb}
          onCreate={handleCreateKb}
          onRename={handleRenameKb}
          onDelete={handleDeleteKb}
        />
        {sessionId && activeKbId && (
          <FileUploadPanel
            sessionId={sessionId}
            kbId={activeKbId}
            files={files}
            onFilesChanged={handleFilesChanged}
          />
        )}
        <button className="reset-btn" onClick={handleResetSession}>
          Reset Session
        </button>
      </aside>
      <main className="main">
        {sessionId && activeKbId ? (
          <ChatWindow
            sessionId={sessionId}
            kbId={activeKbId}
            history={history}
            setHistory={setHistory}
            hasDocuments={files.length > 0}
          />
        ) : (
          <div className="empty">Select or create a knowledge base to start chatting.</div>
        )}
      </main>
    </div>
  );
}
