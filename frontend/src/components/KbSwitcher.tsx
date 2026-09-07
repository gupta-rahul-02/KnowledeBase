import { useState } from "react";
import { KbSummary } from "../api/client";

type Props = {
  kbs: KbSummary[];
  activeKbId: string | null;
  busy?: boolean;
  onSelect: (kbId: string) => void;
  onCreate: (name: string) => Promise<void> | void;
  onRename: (kbId: string, name: string) => Promise<void> | void;
  onDelete: (kbId: string) => Promise<void> | void;
};

export function KbSwitcher({ kbs, activeKbId, busy, onSelect, onCreate, onRename, onDelete }: Props) {
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");

  async function handleCreate() {
    const name = newName.trim();
    if (!name) return;
    await onCreate(name);
    setNewName("");
    setCreating(false);
  }

  async function handleRename(kbId: string) {
    const name = renameValue.trim();
    if (!name) return;
    await onRename(kbId, name);
    setRenamingId(null);
    setRenameValue("");
  }

  return (
    <div className="kb-switcher">
      <h3>Knowledge Bases</h3>
      <ul className="kb-list">
        {kbs.map((kb) => {
          const active = kb.kb_id === activeKbId;
          const isRenaming = renamingId === kb.kb_id;
          return (
            <li key={kb.kb_id} className={`kb-item ${active ? "kb-item-active" : ""}`}>
              {isRenaming ? (
                <input
                  autoFocus
                  className="kb-rename-input"
                  value={renameValue}
                  onChange={(e) => setRenameValue(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleRename(kb.kb_id);
                    else if (e.key === "Escape") setRenamingId(null);
                  }}
                  onBlur={() => setRenamingId(null)}
                />
              ) : (
                <button
                  type="button"
                  className="kb-name"
                  title={kb.name}
                  disabled={busy}
                  onClick={() => onSelect(kb.kb_id)}
                >
                  <span className="kb-name-text">{kb.name}</span>
                  <span className="kb-files-count">{kb.files_count}</span>
                </button>
              )}
              {!isRenaming && (
                <div className="kb-actions">
                  <button
                    type="button"
                    title="Rename"
                    disabled={busy}
                    onClick={() => {
                      setRenamingId(kb.kb_id);
                      setRenameValue(kb.name);
                    }}
                  >
                    ✎
                  </button>
                  <button
                    type="button"
                    title="Delete"
                    disabled={busy || kbs.length <= 1}
                    onClick={() => {
                      if (confirm(`Delete knowledge base "${kb.name}"?`)) onDelete(kb.kb_id);
                    }}
                  >
                    ×
                  </button>
                </div>
              )}
            </li>
          );
        })}
      </ul>
      {creating ? (
        <div className="kb-create">
          <input
            autoFocus
            className="kb-rename-input"
            placeholder="New KB name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleCreate();
              else if (e.key === "Escape") {
                setCreating(false);
                setNewName("");
              }
            }}
          />
          <button type="button" onClick={handleCreate} disabled={!newName.trim() || busy}>
            Add
          </button>
        </div>
      ) : (
        <button type="button" className="kb-new-btn" disabled={busy} onClick={() => setCreating(true)}>
          + New Knowledge Base
        </button>
      )}
    </div>
  );
}
