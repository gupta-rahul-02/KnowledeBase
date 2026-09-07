import { useEffect, useRef, useState } from "react";
import { ChatMessage, Citation, streamChat } from "../api/client";
import { MessageBubble } from "./MessageBubble";

type Props = {
  sessionId: string;
  kbId: string;
  history: ChatMessage[];
  setHistory: (h: ChatMessage[]) => void;
  hasDocuments: boolean;
};

export function ChatWindow({ sessionId, kbId, history, setHistory, hasDocuments }: Props) {
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [history, streaming]);

  async function send() {
    const msg = input.trim();
    if (!msg || streaming) return;
    setInput("");
    setError(null);

    const nextHistory: ChatMessage[] = [
      ...history,
      { role: "user", content: msg },
      { role: "assistant", content: "" },
    ];
    setHistory(nextHistory);
    setStreaming(true);

    let assistantBuffer = "";
    let citations: Citation[] = [];
    await streamChat(sessionId, kbId, msg, 5, {
      onCitations: (c) => {
        citations = c;
        setHistory([
          ...nextHistory.slice(0, -1),
          { role: "assistant", content: assistantBuffer, citations },
        ]);
      },
      onToken: (t) => {
        assistantBuffer += t;
        setHistory([
          ...nextHistory.slice(0, -1),
          { role: "assistant", content: assistantBuffer, citations },
        ]);
      },
      onDone: () => setStreaming(false),
      onError: (m) => {
        setError(m);
        setStreaming(false);
      },
    });
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="chat">
      <div className="chat-messages" ref={scrollRef}>
        {history.length === 0 ? (
          <div className="empty">
            {hasDocuments
              ? "Ask a question about your uploaded documents."
              : "Upload documents in the sidebar, then ask questions here."}
          </div>
        ) : (
          history.map((m, i) => <MessageBubble key={i} msg={m} />)
        )}
      </div>
      {error && <div className="error chat-error">{error}</div>}
      <div className="chat-input">
        <textarea
          placeholder="Ask a question..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          rows={2}
          disabled={streaming}
        />
        <button onClick={send} disabled={streaming || !input.trim()}>
          {streaming ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
}
