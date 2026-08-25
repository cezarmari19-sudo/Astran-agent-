import { storage } from "@/src/utils/storage";

const BASE = process.env.EXPO_PUBLIC_BACKEND_URL;
export const API = `${BASE}/api`;

const TOKEN_KEY = "auth_token";
const DEVICE_ID_KEY = "device_id";

// In-memory cache so every request doesn't hit secure storage; refreshed
// on login/logout and read once at cold start via ensureDeviceId().
let cachedToken: string | null = null;
let sessionExpiredHandler: (() => void) | null = null;

/** Call once near app startup (e.g. in the auth gate) to be notified when
 * any request comes back 401 — used to bounce the user to the login screen
 * without every screen needing its own try/catch for it. */
export function onSessionExpired(handler: () => void) {
  sessionExpiredHandler = handler;
}

export async function getToken(): Promise<string | null> {
  if (cachedToken !== null) return cachedToken;
  const stored = await storage.secureGet(TOKEN_KEY, "");
  cachedToken = (stored as string) || null;
  return cachedToken;
}

async function setToken(token: string | null) {
  cachedToken = token;
  if (token) {
    await storage.secureSet(TOKEN_KEY, token);
  } else {
    await storage.removeItem(TOKEN_KEY);
  }
}

/** A stable per-install identifier, persisted so the backend's per-device
 * login rate limit (5 attempts / 24h) applies consistently across app
 * restarts rather than resetting every launch. Generated without any
 * external crypto/uuid dependency (Math.random() + timestamp is more than
 * sufficient here — this is a rate-limit bucket key, not a security
 * credential, so RFC-4122 compliance or cryptographic randomness isn't
 * required). */
function generateDeviceId(): string {
  const rand = () => Math.floor(Math.random() * 1e9).toString(36);
  return `dev-${Date.now().toString(36)}-${rand()}-${rand()}`;
}

export async function getDeviceId(): Promise<string> {
  const existing = await storage.getItem(DEVICE_ID_KEY, "");
  if (existing) return existing as string;
  const fresh = generateDeviceId();
  await storage.setItem(DEVICE_ID_KEY, fresh);
  return fresh;
}

async function authHeaders(): Promise<Record<string, string>> {
  const token = await getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function req(path: string, opts: RequestInit = {}) {
  const auth = await authHeaders();
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json", ...auth },
    ...opts,
  });
  const text = await res.text();
  let data: any = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (res.status === 401) {
    await setToken(null);
    if (sessionExpiredHandler) sessionExpiredHandler();
  }
  if (!res.ok) {
    const msg = (data && (data.detail || data.message)) || `Eroare ${res.status}`;
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return data;
}

async function uploadZip(path: string, fileUri: string, fileName: string, extraFields?: Record<string, string>) {
  const auth = await authHeaders();
  const form = new FormData();
  form.append("file", {
    uri: fileUri,
    name: fileName,
    type: "application/zip",
  } as any);
  if (extraFields) {
    Object.entries(extraFields).forEach(([k, v]) => form.append(k, v));
  }
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { ...auth },
    body: form as any,
  });
  const text = await res.text();
  let data: any = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (res.status === 401) {
    await setToken(null);
    if (sessionExpiredHandler) sessionExpiredHandler();
  }
  if (!res.ok) {
    const msg = (data && (data.detail || data.message)) || `Eroare ${res.status}`;
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return data;
}

export type ProjFile = { path: string; content: string };
export type Project = {
  id: string;
  name: string;
  description: string;
  files: ProjFile[];
  inspiration_files?: ProjFile[];
  created_at: string;
  updated_at: string;
};
export type AuthUser = { id: string; email: string };

export const api = {
  health: () => req("/health"),

  // ---------------- Auth ----------------
  register: async (email: string, password: string): Promise<{ token: string; user: AuthUser }> => {
    const device_id = await getDeviceId();
    const result = await req("/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password, device_id }),
    });
    await setToken(result.token);
    return result;
  },
  login: async (email: string, password: string): Promise<{ token: string; user: AuthUser }> => {
    const device_id = await getDeviceId();
    const result = await req("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password, device_id }),
    });
    await setToken(result.token);
    return result;
  },
  logout: async () => {
    await setToken(null);
  },
  me: (): Promise<AuthUser> => req("/auth/me"),
  isLoggedIn: async (): Promise<boolean> => {
    const token = await getToken();
    return !!token;
  },

  listProjects: (): Promise<Project[]> => req("/projects"),
  createProject: (name: string, description: string): Promise<Project> =>
    req("/projects", { method: "POST", body: JSON.stringify({ name, description }) }),
  getProject: (id: string): Promise<Project> => req(`/projects/${id}`),
  deleteProject: (id: string) => req(`/projects/${id}`, { method: "DELETE" }),
  getMessages: (id: string) => req(`/projects/${id}/messages`),
  chat: (id: string, message: string, model?: string) =>
    req(`/projects/${id}/chat`, {
      method: "POST",
      body: JSON.stringify({ message, model }),
    }),
  chatStart: (id: string, message: string, model?: string) =>
    req(`/projects/${id}/chat/start`, {
      method: "POST",
      body: JSON.stringify({ message, model }),
    }),
  chatStatus: (jobId: string) => req(`/chat/${jobId}`),
  agentChatStart: (id: string, message: string, model?: string) =>
    req(`/projects/${id}/agent-chat`, {
      method: "POST",
      body: JSON.stringify({ message, model }),
    }),
  agentChatStatus: (jobId: string) => req(`/agent-chat/${jobId}`),
  getModels: () => req("/models"),
  review: (id: string, model?: string) =>
    req(`/projects/${id}/review`, { method: "POST", body: JSON.stringify({ model }) }),
  reviewStatus: (jobId: string) => req(`/review/${jobId}`),
  stop: (id: string) => req(`/projects/${id}/stop?kind=both`, { method: "POST" }),

  uploadProjectZip: (id: string, fileUri: string, fileName: string, mode: "replace" | "merge" = "replace") =>
    uploadZip(`/projects/${id}/upload-zip`, fileUri, fileName, { mode }),
  uploadInspirationZip: (id: string, fileUri: string, fileName: string) =>
    uploadZip(`/projects/${id}/upload-inspiration`, fileUri, fileName),
  clearInspiration: (id: string) => req(`/projects/${id}/inspiration`, { method: "DELETE" }),

  listNotes: () => req("/notes"),
  createNote: (title: string, content: string) =>
    req("/notes", { method: "POST", body: JSON.stringify({ title, content }) }),
  deleteNote: (id: string) => req(`/notes/${id}`, { method: "DELETE" }),

  calculator: (expression: string) =>
    req("/tools/calculator", { method: "POST", body: JSON.stringify({ expression }) }),
  websearch: (query: string) =>
    req("/tools/websearch", { method: "POST", body: JSON.stringify({ query }) }),

  githubRepos: (token: string) =>
    req("/github/repos", { method: "POST", body: JSON.stringify({ token }) }),
  githubCommit: (payload: {
    token: string;
    repo: string;
    branch: string;
    message: string;
    project_id: string;
  }) => req("/github/commit", { method: "POST", body: JSON.stringify(payload) }),
};
