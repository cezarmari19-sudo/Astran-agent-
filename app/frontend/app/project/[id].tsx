import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TextInput,
  Pressable,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
} from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { colors, radius, space } from "@/src/theme";
import { Header, PrimaryButton, useToast } from "@/src/components";
import { api, Project, ProjFile } from "@/src/api";
import { storage } from "@/src/utils/storage";

type Msg = { id: string; role: string; content: string; created_at: string };
type Tab = "chat" | "files" | "review" | "github";

export default function ProjectScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();
  const { show, Toast } = useToast();
  const [project, setProject] = useState<Project | null>(null);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [tab, setTab] = useState<Tab>("chat");
  const scrollRef = useRef<ScrollView>(null);

  const load = useCallback(async () => {
    try {
      const [p, m] = await Promise.all([api.getProject(id!), api.getMessages(id!)]);
      setProject(p);
      setMessages(m);
    } catch (e: any) {
      show(e.message, "err");
    }
  }, [id, show]);

  useEffect(() => {
    load();
  }, [load]);

  const send = async () => {
    const text = input.trim();
    if (!text || sending) return;
    setInput("");
    const optimistic: Msg = {
      id: "tmp-" + Date.now(),
      role: "user",
      content: text,
      created_at: new Date().toISOString(),
    };
    setMessages((m) => [...m, optimistic]);
    setSending(true);
    setTimeout(() => scrollRef.current?.scrollToEnd({ animated: true }), 50);
    try {
      const res = await api.chat(id!, text);
      setMessages((m) => [...m, res.message]);
      if (res.all_files) setProject((p) => (p ? { ...p, files: res.all_files } : p));
    } catch (e: any) {
      show(e.message, "err");
    } finally {
      setSending(false);
      setTimeout(() => scrollRef.current?.scrollToEnd({ animated: true }), 60);
    }
  };

  const tabs: { key: Tab; label: string; icon: any }[] = [
    { key: "chat", label: "Chat", icon: "chatbubble-ellipses" },
    { key: "files", label: "Fișiere", icon: "folder" },
    { key: "review", label: "Review", icon: "shield-checkmark" },
    { key: "github", label: "GitHub", icon: "logo-github" },
  ];

  return (
    <View style={styles.container}>
      <Header
        title={project?.name || "Proiect"}
        subtitle={`${project?.files?.length || 0} fișiere • Gemini 3.1 Pro`}
        onBack={() => router.back()}
      />

      <View style={styles.segment}>
        {tabs.map((t) => (
          <Pressable
            key={t.key}
            testID={`tab-${t.key}`}
            onPress={() => setTab(t.key)}
            style={[styles.segBtn, tab === t.key && styles.segBtnActive]}
          >
            <Ionicons
              name={t.icon}
              size={16}
              color={tab === t.key ? colors.accent : colors.faint}
            />
            <Text style={[styles.segText, tab === t.key && { color: colors.text }]}>
              {t.label}
            </Text>
          </Pressable>
        ))}
      </View>

      {tab === "chat" && (
        <ChatTab
          messages={messages}
          sending={sending}
          input={input}
          setInput={setInput}
          send={send}
          scrollRef={scrollRef}
        />
      )}
      {tab === "files" && <FilesTab files={project?.files || []} />}
      {tab === "review" && (
        <ReviewTab id={id!} onDone={load} show={show} hasFiles={(project?.files?.length || 0) > 0} />
      )}
      {tab === "github" && (
        <GithubTab id={id!} show={show} hasFiles={(project?.files?.length || 0) > 0} />
      )}
      {Toast}
    </View>
  );
}

function ChatTab({ messages, sending, input, setInput, send, scrollRef }: any) {
  return (
    <KeyboardAvoidingView
      style={{ flex: 1 }}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
      keyboardVerticalOffset={90}
    >
      <ScrollView
        ref={scrollRef}
        style={{ flex: 1 }}
        contentContainerStyle={{ padding: space.md, paddingBottom: 20 }}
        onContentSizeChange={() => scrollRef.current?.scrollToEnd({ animated: true })}
      >
        {messages.length === 0 && (
          <View style={styles.hint}>
            <Text style={styles.hintTitle}>Descrie aplicația ta 👇 nu, doar text.</Text>
            <Text style={styles.hintText}>
              Ex: „Fă o aplicație de notițe cu categorii și temă dark, frumoasă și
              modernă.” Aria va planifica și scrie tot codul.
            </Text>
          </View>
        )}
        {messages.map((m: Msg) => (
          <View
            key={m.id}
            testID={`msg-${m.role}`}
            style={[styles.bubble, m.role === "user" ? styles.userBubble : styles.aiBubble]}
          >
            {m.role !== "user" && (
              <View style={styles.aiTag}>
                <View style={styles.brandDot} />
                <Text style={styles.aiTagText}>Aria</Text>
              </View>
            )}
            <Text
              style={[
                styles.bubbleText,
                m.role === "user" && { color: "#04140B" },
              ]}
            >
              {m.content}
            </Text>
          </View>
        ))}
        {sending && (
          <View style={[styles.bubble, styles.aiBubble]}>
            <ActivityIndicator color={colors.accent} />
            <Text style={styles.thinking}>Aria planifică și scrie cod…</Text>
          </View>
        )}
      </ScrollView>
      <View style={styles.inputBar}>
        <TextInput
          testID="chat-input"
          style={styles.chatInput}
          placeholder="Scrie ce vrei să construiască Aria…"
          placeholderTextColor={colors.faint}
          value={input}
          onChangeText={setInput}
          multiline
        />
        <Pressable
          testID="chat-send"
          onPress={send}
          disabled={sending}
          style={({ pressed }) => [styles.sendBtn, pressed && { opacity: 0.8 }]}
        >
          <Ionicons name="arrow-up" size={22} color="#04140B" />
        </Pressable>
      </View>
    </KeyboardAvoidingView>
  );
}

function FilesTab({ files }: { files: ProjFile[] }) {
  const [open, setOpen] = useState<string | null>(null);
  if (files.length === 0)
    return (
      <View style={styles.center}>
        <Ionicons name="folder-open-outline" size={48} color={colors.faint} />
        <Text style={styles.emptyText}>
          Niciun fișier încă. Cere-i Ariei să genereze aplicația în Chat.
        </Text>
      </View>
    );
  return (
    <ScrollView contentContainerStyle={{ padding: space.md, paddingBottom: 40 }}>
      {files.map((f) => (
        <View key={f.path} style={styles.fileCard}>
          <Pressable
            testID={`file-${f.path}`}
            style={styles.fileHead}
            onPress={() => setOpen(open === f.path ? null : f.path)}
          >
            <Ionicons name="document-text-outline" size={18} color={colors.accent2} />
            <Text style={styles.filePath} numberOfLines={1}>
              {f.path}
            </Text>
            <Ionicons
              name={open === f.path ? "chevron-up" : "chevron-down"}
              size={18}
              color={colors.faint}
            />
          </Pressable>
          {open === f.path && (
            <ScrollView horizontal style={styles.codeWrap}>
              <Text style={styles.code}>{f.content}</Text>
            </ScrollView>
          )}
        </View>
      ))}
    </ScrollView>
  );
}

function ReviewTab({ id, onDone, show, hasFiles }: any) {
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<any>(null);
  const pollRef = useRef<any>(null);

  useEffect(() => () => pollRef.current && clearInterval(pollRef.current), []);

  const run = async () => {
    setRunning(true);
    setResult(null);
    try {
      const { job_id } = await api.review(id);
      pollRef.current = setInterval(async () => {
        try {
          const job = await api.reviewStatus(job_id);
          setResult(job);
          if (job.done) {
            clearInterval(pollRef.current);
            setRunning(false);
            onDone();
            if (job.error) show("Eroare la verificare", "err");
            else show(job.stopped_clean ? "Verificare completă — curat!" : "Verificare finalizată");
          }
        } catch {
          /* keep polling */
        }
      }, 2000);
    } catch (e: any) {
      setRunning(false);
      show(e.message, "err");
    }
  };

  return (
    <ScrollView contentContainerStyle={{ padding: space.md, paddingBottom: 40 }}>
      <View style={styles.infoCard}>
        <Text style={styles.infoTitle}>Agent de verificare</Text>
        <Text style={styles.infoText}>
          Un singur agent foarte bun rulează în buclă: caută bug-uri, scurgeri de chei,
          UI generic și le repară. Se oprește după 3 treceri consecutive fără probleme.
        </Text>
      </View>
      <PrimaryButton
        testID="run-review-btn"
        title={running ? "Agentul rulează…" : "Rulează verificarea"}
        icon="shield-checkmark"
        loading={running}
        disabled={!hasFiles || running}
        onPress={run}
      />
      {!hasFiles && <Text style={styles.warnText}>Generează întâi cod în Chat.</Text>}

      {result && (
        <View style={{ marginTop: space.md }}>
          <Text style={styles.sectionTitle}>
            {result.passes.length} treceri{" "}
            {result.done ? (result.stopped_clean ? "• curat ✓" : "• finalizat") : "• în curs…"}
          </Text>
          {result.passes.map((p: any) => (
            <View key={p.pass} style={styles.passCard}>
              <View style={styles.passHead}>
                <Text style={styles.passTitle}>Trecere #{p.pass}</Text>
                <View
                  style={[
                    styles.badge,
                    { backgroundColor: p.issues.length ? colors.warn : colors.accent },
                  ]}
                >
                  <Text style={styles.badgeText}>
                    {p.issues.length ? `${p.issues.length} probleme` : "curat"}
                  </Text>
                </View>
              </View>
              {p.issues.map((iss: any, idx: number) => (
                <View key={idx} style={styles.issue}>
                  <View style={[styles.dot, sevColor(iss.severity)]} />
                  <View style={{ flex: 1 }}>
                    <Text style={styles.issueFile}>{iss.file}</Text>
                    <Text style={styles.issueDesc}>{iss.description}</Text>
                    {iss.fix ? <Text style={styles.issueFix}>Fix: {iss.fix}</Text> : null}
                  </View>
                </View>
              ))}
              {p.summary ? <Text style={styles.passSummary}>{p.summary}</Text> : null}
            </View>
          ))}
          {!result.done && (
            <View style={styles.liveRow}>
              <ActivityIndicator color={colors.accent} />
              <Text style={styles.passSummary}>Agentul continuă verificarea…</Text>
            </View>
          )}
        </View>
      )}
    </ScrollView>
  );
}

function GithubTab({ id, show, hasFiles }: any) {
  const [token, setToken] = useState("");
  const [repo, setRepo] = useState("");
  const [branch, setBranch] = useState("main");
  const [message, setMessage] = useState("Update from AI Builder");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<any>(null);

  useEffect(() => {
    (async () => {
      setToken((await storage.secureGet("gh_token", "")) || "");
      setRepo((await storage.getItem("gh_repo", "")) || "");
    })();
  }, []);

  const commit = async () => {
    if (!token.trim()) return show("Adaugă token-ul GitHub în Setări", "err");
    if (!repo.trim() || !repo.includes("/"))
      return show("Repo trebuie owner/nume", "err");
    setBusy(true);
    setResult(null);
    try {
      await storage.secureSet("gh_token", token.trim());
      await storage.setItem("gh_repo", repo.trim());
      const res = await api.githubCommit({
        token: token.trim(),
        repo: repo.trim(),
        branch: branch.trim() || "main",
        message: message.trim() || "Update",
        project_id: id,
      });
      setResult(res);
      show(`${res.committed}/${res.total} fișiere trimise`);
    } catch (e: any) {
      show(e.message, "err");
    } finally {
      setBusy(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={{ padding: space.md, paddingBottom: 40 }}>
      <Text style={styles.label}>Repository (owner/nume)</Text>
      <TextInput
        testID="gh-repo-input"
        style={styles.input}
        placeholder="utilizator/repo"
        placeholderTextColor={colors.faint}
        autoCapitalize="none"
        value={repo}
        onChangeText={setRepo}
      />
      <Text style={styles.label}>Branch</Text>
      <TextInput
        testID="gh-branch-input"
        style={styles.input}
        placeholder="main"
        placeholderTextColor={colors.faint}
        autoCapitalize="none"
        value={branch}
        onChangeText={setBranch}
      />
      <Text style={styles.label}>Mesaj commit</Text>
      <TextInput
        testID="gh-message-input"
        style={styles.input}
        placeholderTextColor={colors.faint}
        value={message}
        onChangeText={setMessage}
      />
      <Text style={styles.label}>Token (salvat securizat)</Text>
      <TextInput
        testID="gh-token-input"
        style={styles.input}
        placeholder="ghp_..."
        placeholderTextColor={colors.faint}
        autoCapitalize="none"
        secureTextEntry
        value={token}
        onChangeText={setToken}
      />
      <View style={{ height: space.md }} />
      <PrimaryButton
        testID="gh-commit-btn"
        title="Commit pe GitHub"
        icon="logo-github"
        loading={busy}
        disabled={!hasFiles}
        onPress={commit}
      />
      {result && (
        <View style={{ marginTop: space.md }}>
          {result.results.map((r: any) => (
            <View key={r.path} style={styles.commitRow}>
              <Ionicons
                name={r.ok ? "checkmark-circle" : "close-circle"}
                size={16}
                color={r.ok ? colors.accent : colors.danger}
              />
              <Text style={styles.commitPath} numberOfLines={1}>
                {r.path}