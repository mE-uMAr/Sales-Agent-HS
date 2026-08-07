"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import {
  ApiError,
  type HandoffReason,
  type Stage,
  apiBase,
  checkHealth,
  closeSession,
  sendMessage,
  startSession,
} from "@/lib/api";

type Role = "visitor" | "assistant" | "system";

interface Turn {
  id: number;
  role: Role;
  text: string;
}

interface Session {
  id: string;
  token: string;
}

type Phase = "form" | "chatting" | "closed";

const HANDOFF_LABELS: Record<HandoffReason, string> = {
  completed: "Wrapped up normally",
  user_requested: "Visitor asked for a person",
  agent_escalated: "Assistant escalated",
  knowledge_gap: "Too many unanswerable questions",
  max_turns: "Turn limit reached",
  idle_timeout: "Abandoned",
  error: "Ended on an error",
};

let turnId = 0;
const nextId = () => ++turnId;

export default function Chat() {
  const [phase, setPhase] = useState<Phase>("form");
  const [session, setSession] = useState<Session | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [stage, setStage] = useState<Stage>("greeting");
  const [handoff, setHandoff] = useState<HandoffReason | null>(null);
  const [pending, setPending] = useState(false);
  const [backendUp, setBackendUp] = useState<boolean | null>(null);

  const [name, setName] = useState("Dana Reyes");
  const [email, setEmail] = useState("dana@brightpath.io");
  const [company, setCompany] = useState("Brightpath");
  const [phone, setPhone] = useState("");
  const [draft, setDraft] = useState("");

  const scrollRef = useRef<HTMLDivElement>(null);
  const sessionRef = useRef<Session | null>(null);
  sessionRef.current = session;

  useEffect(() => {
    void checkHealth().then(setBackendUp);
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [turns, pending]);

  // Best-effort close so a visitor who navigates away still becomes a lead.
  // The idle sweeper catches the rest, so this failing is not a problem.
  useEffect(() => {
    const onUnload = () => {
      const active = sessionRef.current;
      if (active) void closeSession(active.id, active.token).catch(() => {});
    };
    window.addEventListener("beforeunload", onUnload);
    return () => window.removeEventListener("beforeunload", onUnload);
  }, []);

  const append = useCallback((role: Role, text: string) => {
    setTurns((prev) => [...prev, { id: nextId(), role, text }]);
  }, []);

  const handleFailure = useCallback(
    (error: unknown) => {
      if (error instanceof ApiError) {
        append("system", `${error.status} — ${error.message}`);
        if (error.isTerminal) {
          setPhase("closed");
          setSession(null);
        }
        return;
      }
      append(
        "system",
        `Could not reach the backend at ${apiBase}. Is it running?`,
      );
      setBackendUp(false);
    },
    [append],
  );

  async function handleStart(event: React.FormEvent) {
    event.preventDefault();
    if (!name.trim() || pending) return;

    setPending(true);
    setTurns([]);
    setHandoff(null);

    try {
      const started = await startSession({
        name: name.trim(),
        email: email.trim() || undefined,
        company: company.trim() || undefined,
        phone: phone.trim() || undefined,
        utm: { utm_source: "test-harness" },
      });
      setSession({ id: started.session_id, token: started.token });
      setStage(started.stage);
      setPhase("chatting");
      append("assistant", started.message);
    } catch (error) {
      handleFailure(error);
    } finally {
      setPending(false);
    }
  }

  async function handleSend(event?: React.FormEvent) {
    event?.preventDefault();
    const text = draft.trim();
    if (!text || !session || pending) return;

    setDraft("");
    append("visitor", text);
    setPending(true);

    try {
      const result = await sendMessage(session.id, session.token, text);
      append("assistant", result.reply);
      setStage(result.stage);

      if (result.closed) {
        setHandoff(result.handoff_reason);
        setPhase("closed");
        setSession(null);
      }
    } catch (error) {
      handleFailure(error);
    } finally {
      setPending(false);
    }
  }

  async function handleClose() {
    if (!session || pending) return;
    setPending(true);
    try {
      const result = await closeSession(session.id, session.token);
      append(
        "system",
        result.lead_captured
          ? "Conversation closed — lead captured."
          : "Conversation was already closed; no duplicate lead written.",
      );
      setHandoff("completed");
    } catch (error) {
      handleFailure(error);
    } finally {
      setPhase("closed");
      setSession(null);
      setPending(false);
    }
  }

  function restart() {
    setPhase("form");
    setSession(null);
    setTurns([]);
    setStage("greeting");
    setHandoff(null);
    setDraft("");
  }

  return (
    <>
      <div className="masthead">
        <h1>Hashed Assistant</h1>
        <span className="status">
          <span
            className={`dot ${backendUp === null ? "" : backendUp ? "up" : "down"}`}
          />
          {backendUp === null
            ? "checking backend…"
            : backendUp
              ? apiBase
              : `unreachable — ${apiBase}`}
        </span>
      </div>
      <p className="subtitle">
        Test harness for the sales assistant API. The WordPress widget will
        replace this page; the calls it makes are in{" "}
        <code>lib/api.ts</code>.
      </p>

      {phase === "form" ? (
        <div className="panel">
          <div className="panel-head">
            <span className="panel-title">
              Contact details — what the form used to collect
            </span>
          </div>
          <form className="form" onSubmit={handleStart}>
            <div className="grid-2">
              <div className="field">
                <label htmlFor="name">Name (required)</label>
                <input
                  id="name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required
                  maxLength={200}
                />
              </div>
              <div className="field">
                <label htmlFor="email">Email</label>
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </div>
            </div>
            <div className="grid-2">
              <div className="field">
                <label htmlFor="company">Company</label>
                <input
                  id="company"
                  value={company}
                  onChange={(e) => setCompany(e.target.value)}
                />
              </div>
              <div className="field">
                <label htmlFor="phone">Phone</label>
                <input
                  id="phone"
                  value={phone}
                  onChange={(e) => setPhone(e.target.value)}
                />
              </div>
            </div>
            <div>
              <button
                type="submit"
                className="btn-primary"
                disabled={pending || !name.trim()}
              >
                {pending ? "Starting…" : "Start conversation"}
              </button>
            </div>
          </form>
        </div>
      ) : (
        <div className="panel">
          <div className="panel-head">
            <span className="panel-title">
              Conversation <span className="tag">{stage}</span>
            </span>
            {phase === "chatting" ? (
              <button
                className="btn-ghost"
                onClick={handleClose}
                disabled={pending}
              >
                End &amp; capture lead
              </button>
            ) : (
              <button className="btn-ghost" onClick={restart}>
                New conversation
              </button>
            )}
          </div>

          <div className="transcript" ref={scrollRef}>
            {turns.map((turn) => (
              <div key={turn.id} className={`turn ${turn.role}`}>
                <div className="bubble">{turn.text}</div>
              </div>
            ))}
            {pending && (
              <div className="turn assistant">
                <div className="bubble">
                  <span className="typing">
                    <span />
                    <span />
                    <span />
                  </span>
                </div>
              </div>
            )}
          </div>

          {phase === "chatting" ? (
            <form className="composer" onSubmit={handleSend}>
              <textarea
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void handleSend();
                  }
                }}
                placeholder="Ask about services, past work, or pricing…"
                maxLength={2000}
                rows={1}
                disabled={pending}
              />
              <button
                type="submit"
                className="btn-primary"
                disabled={pending || !draft.trim()}
              >
                Send
              </button>
            </form>
          ) : (
            <div className="closed-note">
              <span>
                Conversation ended
                {handoff ? ` — ${HANDOFF_LABELS[handoff]}` : ""}. The lead is in
                the database.
              </span>
              <button className="btn-ghost" onClick={restart}>
                Start again
              </button>
            </div>
          )}
        </div>
      )}

      {session && (
        <div className="debug">
          <span>
            <b>session</b> {session.id.slice(0, 8)}…
          </span>
          <span>
            <b>stage</b> {stage}
          </span>
          <span>
            <b>turns</b> {turns.filter((t) => t.role === "visitor").length}
          </span>
        </div>
      )}

      <p className="hint">
        Try: <em>“We need a customer portal”</em> → <em>“How much would that
        cost?”</em> → <em>“Our budget is around 20 to 30 thousand”</em> →{" "}
        <em>“Who founded the company?”</em> (the last one should be answered
        honestly with “I don’t know”). Inspect the resulting lead with{" "}
        <code>
          curl {apiBase}/v1/admin/leads -H &quot;X-Admin-Key: …&quot;
        </code>
        .
      </p>
    </>
  );
}
