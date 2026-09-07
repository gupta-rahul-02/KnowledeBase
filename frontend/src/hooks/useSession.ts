import { useEffect, useState } from "react";
import { createSession, getSession, SessionInfo, setActiveKb } from "../api/client";

const SESSION_KEY = "kb.session_id";
const ACTIVE_KB_KEY = "kb.active_kb_id";

export function useSession() {
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const storedSession = localStorage.getItem(SESSION_KEY);
    const storedKb = localStorage.getItem(ACTIVE_KB_KEY);

    async function boot() {
      try {
        let info: SessionInfo | null = null;
        if (storedSession) {
          try {
            info = await getSession(storedSession);
          } catch {
            localStorage.removeItem(SESSION_KEY);
            localStorage.removeItem(ACTIVE_KB_KEY);
          }
        }
        if (!info) {
          info = await createSession();
          localStorage.setItem(SESSION_KEY, info.session_id);
        }
        // Reconcile stored active KB with server state
        if (storedKb && info.kbs.some((k) => k.kb_id === storedKb) && storedKb !== info.active_kb_id) {
          try {
            await setActiveKb(info.session_id, storedKb);
            info = { ...info, active_kb_id: storedKb };
          } catch {
            /* ignore */
          }
        }
        if (info.active_kb_id) {
          localStorage.setItem(ACTIVE_KB_KEY, info.active_kb_id);
        }
        if (!cancelled) setSession(info);
      } catch (e) {
        if (!cancelled) setError(String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    boot();
    return () => {
      cancelled = true;
    };
  }, []);

  function persistActiveKb(kbId: string | null) {
    if (kbId) localStorage.setItem(ACTIVE_KB_KEY, kbId);
    else localStorage.removeItem(ACTIVE_KB_KEY);
  }

  function resetLocalSession() {
    localStorage.removeItem(SESSION_KEY);
    localStorage.removeItem(ACTIVE_KB_KEY);
    setSession(null);
  }

  return { session, setSession, loading, error, persistActiveKb, resetLocalSession };
}
