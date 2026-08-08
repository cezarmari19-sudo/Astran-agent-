const BASE = process.env.EXPO_PUBLIC_BACKEND_URL;
export const API = `${BASE}/api`;

async function req(path: string, opts: RequestInit = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  const text = await res.text();
  let data: any = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
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
  created_at: string;
  updated_at: string;
};

export const api = {
  health: () => req("/health"),

  listProjects: (): Promise<Project[]> => req("/projects"),
  createProject: (name: string, description: string): Promise<Project> =>
    req("/projects", { method: "POST", body: JSON.stringify({ name, description }) }),
  getProject: (id: string): Promise<Project> => req(`/projects/${id}`),
  deleteProject: (id: string) => req(`/projects/${id}`, { method: "DELETE" }),
  getMessages: (id: string) => req(`/projects/${id}/messages`),
  chat: (id: string, message: string) =>
    req(`/projects/${id}/chat`, { method: "POST", body: JSON.stringify({ message }) }),
  review: (id: string) => req(`/projects/${id}/review`, { method: "POST" }),

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