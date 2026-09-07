import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ChatMessage } from "../api/client";

export function MessageBubble({ msg }: { msg: ChatMessage }) {
  const showCitations =
    msg.role === "assistant" && msg.citations && msg.citations.length > 0;
  return (
    <div className={`bubble bubble-${msg.role}`}>
      <div className="bubble-role">{msg.role === "user" ? "You" : "Assistant"}</div>
      <div className="bubble-content">
        {!msg.content ? (
          <span className="muted">...</span>
        ) : msg.role === "assistant" ? (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
        ) : (
          msg.content
        )}
      </div>
      {showCitations && (
        <div className="bubble-citations">
          <span className="citations-label">Sources:</span>{" "}
          {msg.citations!.map((c, i) => (
            <span key={`${c.source}-${c.page ?? ""}-${i}`} className="citation-chip">
              {c.source}
              {c.page ? ` (p.${c.page})` : ""}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
