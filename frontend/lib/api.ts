/**
 * Client for the Hashed RAG assistant API.
 *
 * This file is the part worth copying into the WordPress widget — it is plain
 * `fetch` with no framework dependency, and it encodes the whole contract:
 * three calls, one token, and the `closed` flag that ends the conversation.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
const WIDGET_KEY = process.env.NEXT_PUBLIC_WIDGET_KEY ?? "pub_dev";

export interface ContactPayload {
  name: string;
  email?: string;
  phone?: string;
  company?: string;
  page_url?: string;
  utm?: Record<string, string>;
}

export interface StartSessionResponse {
  session_id: string;
  token: string;
  expires_in: number;
  /** Canned opening line — render it as the first assistant bubble. */
  message: string;
  stage: Stage;
}

export interface MessageResponse {
  session_id: string;
  reply: string;
  stage: Stage;
  /** When true, stop sending messages. The lead has been captured. */
  closed: boolean;
  handoff_reason: HandoffReason | null;
}

export interface CloseResponse {
  session_id: string;
  closed: boolean;
  lead_captured: boolean;
  message: string;
}

export type Stage = "greeting" | "discovery" | "qualify" | "wrap_up" | "handoff";

export type HandoffReason =
  | "completed"
  | "user_requested"
  | "agent_escalated"
  | "knowledge_gap"
  | "max_turns"
  | "idle_timeout"
  | "error";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }

  /** The conversation cannot continue; the UI should offer a fresh start. */
  get isTerminal(): boolean {
    return this.status === 401 || this.status === 404 || this.status === 409;
  }

  get isRateLimit(): boolean {
    return this.status === 429;
  }
}

async function parse<T>(response: Response): Promise<T> {
  if (response.ok) {
    return (await response.json()) as T;
  }

  let detail = response.statusText;
  try {
    const body = await response.json();
    detail = body?.detail ?? body?.error ?? detail;
    if (Array.isArray(detail)) {
      detail = detail.map((d: { msg?: string }) => d.msg ?? "invalid").join(", ");
    }
  } catch {
    // A non-JSON error body is fine — the status carries the meaning.
  }

  throw new ApiError(response.status, String(detail));
}

export async function startSession(
  contact: ContactPayload,
): Promise<StartSessionResponse> {
  const response = await fetch(`${API_BASE}/v1/sessions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Widget-Key": WIDGET_KEY,
    },
    body: JSON.stringify({
      ...contact,
      page_url:
        contact.page_url ??
        (typeof window !== "undefined" ? window.location.href : undefined),
    }),
  });
  return parse<StartSessionResponse>(response);
}

export async function sendMessage(
  sessionId: string,
  token: string,
  message: string,
): Promise<MessageResponse> {
  const response = await fetch(`${API_BASE}/v1/sessions/${sessionId}/messages`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ message }),
  });
  return parse<MessageResponse>(response);
}

export async function closeSession(
  sessionId: string,
  token: string,
): Promise<CloseResponse> {
  const response = await fetch(`${API_BASE}/v1/sessions/${sessionId}/close`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    keepalive: true,
  });
  return parse<CloseResponse>(response);
}

/** Health probe, used by the page to tell "backend down" from "bad request". */
export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/health`, { cache: "no-store" });
    return response.ok;
  } catch {
    return false;
  }
}

export const apiBase = API_BASE;
