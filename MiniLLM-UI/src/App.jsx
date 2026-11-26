import { useEffect, useRef, useState } from "react";
import "./App.css";

const STORAGE_KEY = "llm-chats";

// Next free number, so names don't repeat among existing chats
function nextName(chats) {
  const used = chats.map((c) => Number(/^Chat (\d+)$/.exec(c.name)?.[1] ?? 0));
  return `Chat ${Math.max(0, ...used) + 1}`;
}

const newChat = (name) => ({
  id: `${Date.now()}-${Math.random()}`,
  name,
  messages: [],
});

function loadChats() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (Array.isArray(saved) && saved.length) {
      // Also converts chats saved by the old version (oldest is last in the list)
      return saved.map((c, i) =>
        c.messages ? c : {
          id: c.id,
          name: `Chat ${saved.length - i}`,
          messages: c.text ? [{ role: "model", text: c.text }] : [],
        }
      );
    }
  } catch {
    // unreadable storage: start fresh
  }
  return [newChat("Chat 1")];
}

export default function App() {
  const [chats, setChats] = useState(loadChats);
  const [activeId, setActiveId] = useState(() => chats[0].id);
  const [maxTokens, setMaxTokens] = useState(100);
  const [temperature, setTemperature] = useState(0.8);
  const [loadingId, setLoadingId] = useState(null);
  const [error, setError] = useState("");
  const [draft, setDraft] = useState("");
  const [editingId, setEditingId] = useState(null);
  const [editName, setEditName] = useState("");
  const endRef = useRef(null);

  const active = chats.find((c) => c.id === activeId) ?? chats[0];
  const busy = loadingId !== null;

  // Save automatically whenever chats change
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
    } catch {
      // storage full or blocked
    }
  }, [chats]);

  // Scroll to the newest message
  useEffect(() => {
    endRef.current?.scrollIntoView();
  }, [activeId, active.messages.length, loadingId]);

      const history = [...active.messages.map((m) => m.text), prompt]
      .filter((text) => text !== "(no output)")
      .join("\n");
  function addMessage(id, message) {
    setChats((prev) =>
      prev.map((c) => (c.id === id ? { ...c, messages: [...c.messages, message] } : c))
    );
  }

  function startRename(chat) {
    setEditingId(chat.id);
    setEditName(chat.name);
  }

  function saveRename() {
    const name = editName.trim();
    if (name) {
      setChats((prev) => prev.map((c) => (c.id === editingId ? { ...c, name } : c)));
    }
    setEditingId(null);
  }

  function selectChat(id) {
    setActiveId(id);
    setError("");
    setDraft("");
  }

  function addChat() {
    const chat = newChat(nextName(chats));
    setChats((prev) => [chat, ...prev]);
    selectChat(chat.id);
  }

  function deleteChat(id) {
    if (!window.confirm("Delete this chat?")) return;
    const rest = chats.filter((c) => c.id !== id);
    const next = rest.length ? rest : [newChat("Chat 1")];
    setChats(next);
    if (id === active.id) selectChat(next[0].id);
  }

  async function generate() {
    const prompt = draft.trim();
    if (!prompt || busy) return;
    const { id } = active;
    addMessage(id, { role: "user", text: prompt });
    setDraft("");
    setLoadingId(id);
    setError("");
    try {
      const res = await fetch("http://localhost:8000/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          max_new_tokens: maxTokens,
          temperature,
        }),
      });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const data = await res.json();
      // <|endoftext|> marks the end of a document, so drop anything after it
      const reply = data.text.split("<|endoftext|>")[0].trim();
      addMessage(id, { role: "model", text: reply || "(no output)" });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoadingId(null);
    }
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <button className="new-chat" onClick={addChat}>+ New chat</button>
        <p className="sidebar-label">Saved chats</p>
        <ul>
          {chats.map((c) => (
            <li key={c.id} className={c.id === active.id ? "active" : ""}>
              {editingId === c.id ? (
                <input
                  className="rename-input"
                  value={editName}
                  autoFocus
                  aria-label="Chat name"
                  onChange={(e) => setEditName(e.target.value)}
                  onBlur={saveRename}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") saveRename();
                    if (e.key === "Escape") setEditingId(null);
                  }}
                />
              ) : (
                <>
                  <button className="chat-title" onClick={() => selectChat(c.id)}>
                    {c.name}
                  </button>
                  <button className="rename" onClick={() => startRename(c)} aria-label="Rename chat">
                    ✎
                  </button>
                </>
              )}
              <button
                className="delete"
                onClick={() => deleteChat(c.id)}
                disabled={c.id === loadingId}
                aria-label="Delete chat"
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      </aside>

      <main className="main">
        <header>
          <h1>MiniLLM</h1>
          <span className="badge">Base model</span>
        </header>

        <div className="messages">
          {active.messages.length === 0 && (
            <p className="empty">Start a sentence and the model will continue it.</p>
          )}
          {active.messages.map((m, i) => (
            <div key={i} className={`message ${m.role}`}>{m.text}</div>
          ))}
          {loadingId === active.id && <div className="message model">Generating...</div>}
          <div ref={endRef} />
        </div>

        {error && <p className="error">{error}</p>}

        <div className="composer">
          <textarea
            rows={2}
            placeholder="Type a prompt..."
            aria-label="Prompt"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                generate();
              }
            }}
          />
          <button className="generate" onClick={generate}
            disabled={busy || !draft.trim()}>
            {busy ? "Generating..." : "Send"}
          </button>
        </div>

        <div className="controls">
          <label>
            <span>Max new tokens <b>{maxTokens}</b></span>
            <input type="range" min={10} max={500} step={10} value={maxTokens}
              onChange={(e) => setMaxTokens(Number(e.target.value))} />
          </label>
          <label>
            <span>Temperature <b>{temperature}</b></span>
            <input type="range" min={0.1} max={2} step={0.1} value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))} />
          </label>
        </div>
      </main>
    </div>
  );
}
