import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TextInput,
  Pressable,
  Modal,
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

type ClarifyQuestion = { question: string; options: string[] };
type Msg = {
  id: string;
  role: string;
  content: string;
  created_at: string;
  msg_type?: "normal" | "clarify" | "clarify_answer";
  questions?: ClarifyQuestion[];
};
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
  const [confirmDel, setConfirmDel] = useState(false);
  const [model, setModel] = useState("gemini-3.5-flash");
  const [models, setModels] = useState<any[]>([]);
  const [providersAvailable, setProvidersAvailable] = useState<any>({});
  const [emergentFallback, setEmergentFallback] = useState(false);
  const [modelSheet, setModelSheet] = useState(false);
  const [stopping, setStopping] = useState(false);
  const scrollRef = useRef<ScrollView>(null);

  const removeProject = async () => {
    try {
      await api.deleteProject(id!);
      router.back();
    } catch (e: any) {
      show(e.message, "err");
    }
  };

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

  useEffect(() => {
    (async () => {
      const saved = await storage.getItem("chat_model", "");
      if (saved) setModel(saved as string);
      try {
        const m = await api.getModels();
        setModels(m.models);
        setProvidersAvailable(m.providers_available || {});
        setEmergentFallback(!!m.emergent_fallback);
      } catch {}
    })();
  }, []);

  const pickModel = async (id: string) => {
    setModel(id);
    setModelSheet(false);
    await storage.setItem("chat_model", id);
  };

  const sendText = async (text: string) => {
    if (!text.trim() || sending) return;
    setInput("");
    const optimistic: Msg = {
      id: "tmp-" + Date.now(),
      role: "user",
      content: text,
      created_at: new Date().toISOString(),
      msg_type: "normal",
    };
    setMessages((m) => [...m, optimistic]);
    setSending(true);
    setTimeout(() => scrollRef.current?.scrollToEnd({ animated: true }), 50);
    try {
      const res = await api.chat(id!, text, model);
      if (res.stopped) {
        setMessages((m) => m.filter((msg) => msg.id !== optimistic.id));
        show("Oprit", "err");
      } else {
        setMessages((m) => [...m, res.message]);
        if (res.all_files) setProject((p) => (p ? { ...p, files: res.all_files } : p));
      }
    } catch (e: any) {
      show(e.message, "err");
    } finally {
      setSending(false);
      setTimeout(() => scrollRef.current?.scrollToEnd({ animated: true }), 60);
    }
  };

  const send = () => sendText(input);
  const pickAnswer = (text: string) => sendText(text);

  const stopEverything = async () => {
    setStopping(true);
    try {
      await api.stop(id!);
      show("Oprit");
    } catch (e: any) {
      show(e.message, "err");
    } finally {
      setStopping(false);
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
        subtitle={`${project?.files?.length || 0} fișiere • Gemini 3.5 Flash`}
        onBack={() => router.back()}
        right={
          <Pressable testID="delete-chat-btn" onPress={() => setConfirmDel(true)} hitSlop={10}>
            <Ionicons name="trash-outline" size={22} color={colors.danger} />
          </Pressable>
        }
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
          onPickAnswer={pickAnswer}
          scrollRef={scrollRef}
          model={model}
          models={models}
          providersAvailable={providersAvailable}
          emergentFallback={emergentFallback}
          showModels={modelSheet}
          onToggleModels={() => setModelSheet((s: boolean) => !s)}
          onPick={pickModel}
          onStop={stopEverything}
          stopping={stopping}
        />
      )}
      {tab === "files" && <FilesTab files={project?.files || []} />}
      {tab === "review" && (
        <ReviewTab
          id={id!}
          onDone={load}
          show={show}
          hasFiles={(project?.files?.length || 0) > 0}
          model={model}
          onStop={stopEverything}
          stopping={stopping}
        />
      )}
      {tab === "github" && (
        <GithubTab id={id!} show={show} hasFiles={(project?.files?.length || 0) > 0} />
      )}

      <Modal visible={confirmDel} transparent animationType="fade" onRequestClose={() => setConfirmDel(false)}>
        <View style={styles.confirmWrap}>
          <View style={styles.confirmCard}>
            <Ionicons name="trash" size={30} color={colors.danger} />
            <Text style={styles.confirmTitle}>Ștergi acest chat?</Text>
            <Text style={styles.confirmText}>
              Conversația și tot codul generat se șterg definitiv.
            </Text>
            <View style={styles.confirmRow}>
              <Pressable style={[styles.confirmBtn, styles.cancelBtn]} onPress={() => setConfirmDel(false)}>
                <Text style={styles.cancelText}>Anulează</Text>
              </Pressable>
              <Pressable testID="confirm-delete-chat" style={[styles.confirmBtn, styles.delBtn]} onPress={removeProject}>
                <Text style={styles.delText}>Șterge</Text>
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>
      {Toast}
    </View>
  );
}

function ChatTab({
  messages,
  sending,
  input,
  setInput,
  send,
  onPickAnswer,
  scrollRef,
  model,
  models,
  providersAvailable,
  emergentFallback,
  showModels,
  onToggleModels,
  onPick,
  onStop,
  stopping,
}: any) {
  const lastMsg = messages[messages.length - 1];
  const awaitingClarification = lastMsg && lastMsg.msg_type === "clarify" && !sending;

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
          <View key={m.id}>
            <View
              testID={`msg-${m.role}`}
              style={[
                styles.bubble,
                m.role === "user" ? styles.userBubble : styles.aiBubble,
                m.msg_type === "clarify" && styles.clarifyBubble,
              ]}
            >
              {m.role !== "user" && (
                <View style={styles.aiTag}>
                  <View style={styles.brandDot} />
                  <Text style={styles.aiTagText}>
                    {m.msg_type === "clarify" ? "Aria — câteva detalii" : "Aria"}
                  </Text>
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

            {m.msg_type === "clarify" && Array.isArray(m.questions) && (
              <View style={styles.clarifyBlock}>
                {m.questions.map((q: ClarifyQuestion, qi: number) => (
                  <View key={qi} style={styles.clarifyQuestion}>
                    <Text style={styles.clarifyQuestionText}>{q.question}</Text>
                    <View style={styles.clarifyOptions}>
                      {q.options.map((opt: string, oi: number) => (
                        <Pressable
                          key={oi}
                          testID={`clarify-opt-${qi}-${oi}`}
                          onPress={() => onPickAnswer(opt)}
                          style={({ pressed }) => [
                            styles.clarifyOptBtn,
                            pressed && { opacity: 0.7 },
                          ]}
                        >
                          <Text style={styles.clarifyOptText}>
                            {oi + 1}. {opt}
                          </Text>
                        </Pressable>
                      ))}
                    </View>
                  </View>
                ))}
                <Text style={styles.clarifyFreeHint}>
                  Poți alege una din opțiuni sau scrie tu ce vrei mai jos.
                </Text>
              </View>
            )}
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
        {showModels && (
          <ScrollView style={styles.modelPanel} keyboardShouldPersistTaps="handled">
            <Text style={styles.modelPanelTitle}>Alege modelul AI</Text>
            {(models || []).map((m: any) => {
              const hasOwnKey = providersAvailable?.[m.provider];
              const usable = hasOwnKey || emergentFallback;
              return (
                <Pressable
                  key={m.id}
                  testID={`model-${m.id}`}
                  onPress={() => usable && onPick(m.id)}
                  disabled={!usable}
                  style={[
                    styles.modelRow,
                    model === m.id && styles.modelRowActive,
                    !usable && styles.modelRowDisabled,
                  ]}
                >
                  <View style={{ flex: 1 }}>
                    <Text style={[styles.modelLabel, !usable && styles.modelLabelDisabled]}>
                      {m.label}
                    </Text>
                    <Text style={styles.modelHint}>
                      {usable ? m.hint : `Necesită cheie ${m.provider} — adaugă în Setări`}
                    </Text>
                  </View>
                  {model === m.id && usable && (
                    <Ionicons name="checkmark-circle" size={18} color={colors.accent} />
                  )}
                </Pressable>
              );
            })}
          </ScrollView>
        )}
        <View style={styles.pillRow}>
          <Pressable testID="model-pill" onPress={onToggleModels} style={styles.modelPill}>
            <Ionicons name="flash" size={14} color={colors.accent} />
            <Text style={styles.modelPillText} numberOfLines={1}>
              {model}
            </Text>
            <Ionicons name={showModels ? "chevron-down" : "chevron-up"} size={13} color={colors.faint} />
          </Pressable>
          {sending && (
            <Pressable testID="stop-btn" onPress={onStop} disabled={stopping} style={styles.stopPill}>
              <Ionicons name="stop-circle" size={14} color={colors.danger} />
              <Text style={styles.stopPillText}>{stopping ? "Oprire…" : "Stop"}</Text>
            </Pressable>
          )}
        </View>
        {awaitingClarification && (
          <View style={styles.clarifyNudge}>
            <Ionicons name="help-circle-outline" size={14} color={colors.accent2} />
            <Text style={styles.clarifyNudgeText}>
              Alege o opțiune de mai sus sau scrie răspunsul tău
            </Text>
          </View>
        )}
        <View style={styles.inputRow}>
          <TextInput
            testID="chat-input"
            style={styles.chatInput}
            placeholder={
              awaitingClarification
                ? "Scrie răspunsul tău…"
                : "Scrie ce vrei să construiască Aria…"
            }
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

function ReviewTab({ id, onDone, show, hasFiles, model, onStop, stopping }: any) {
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<any>(null);
  const pollRef = useRef<any>(null);

  useEffect(() => () => pollRef.current && clearInterval(pollRef.current), []);

  const run = async () => {
    setRunning(true);
    setResult(null);
    try {
      const { job_id } = await api.review(id, model);
      pollRef.current = setInterval(async () => {
        try {
          const job = await api.reviewStatus(job_id);
          setResult(job);
          if (job.done) {
            clearInterval(pollRef.current);
            setRunning(false);
            onDone();
            if (job.error) show("Eroare la verificare", "err");
            else if (job.phase === "stopped") show("Verificare oprită");
            else show("Verificare completă — curat!");
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
        <Text style={styles.infoTitle}>8 agenți de verificare</Text>
        <Text style={styles.infoText}>
          Fiecare rulare de cod pornește automat 8 agenți specializați (bug-uri, design,
          arată uman, user normal, secrete server, code review static, cheat, hacker) —
          rulează pe rând, în buclă, până fiecare confirmă de 2 ori la rând că e curat.
        </Text>
      </View>
      <View style={{ flexDirection: "row", gap: space.sm }}>
        <View style={{ flex: 1 }}>
          <PrimaryButton
            testID="run-review-btn"
            title={running ? "Agenții rulează…" : "Rulează verificarea"}
            icon="shield-checkmark"
            loading={running}
            disabled={!hasFiles || running}
            onPress={run}
          />
        </View>
        {running && (
          <Pressable testID="review-stop-btn" onPress={onStop} disabled={stopping} style={styles.stopBtn}>
            <Ionicons name="stop-circle" size={20} color={colors.danger} />
          </Pressable>
        )}
      </View>
      {!hasFiles && <Text style={styles.warnText}>Generează întâi cod în Chat.</Text>}

      {result && (
        <View style={{ marginTop: space.md }}>
          <Text style={styles.sectionTitle}>
            {result.total_passes} treceri • faza: {result.phase}
          </Text>
          {result.passes.map((p: any, idx: number) => (
            <View key={idx} style={styles.passCard}>
              <View style={styles.passHead}>
                <Text style={styles.passTitle}>{p.label}</Text>
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
              {p.issues.map((iss: any, i2: number) => (
                <View key={i2} style={styles.issue}>
                  <View style={[styles.dot, sevColor(iss.severity)]} />
                  <View style={{ flex: 1 }}>
                    <Text style={styles.issueFile} numberOfLines={1}>
                      {iss.file}
                    </Text>
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
              <Text style={styles.passSummary}>Agenții continuă verificarea…</Text>
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
              </Text>
              {!r.ok && <Text style={styles.commitErr}>{r.error}</Text>}
            </View>
          ))}
        </View>
      )}
    </ScrollView>
  );
}

function sevColor(sev: string) {
  if (sev === "high") return { backgroundColor: colors.danger };
  if (sev === "medium") return { backgroundColor: colors.warn };
  return { backgroundColor: colors.accent2 };
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  center: { flex: 1, alignItems: "center", justifyContent: "center", padding: space.lg, gap: 10 },
  emptyText: { color: colors.muted, textAlign: "center", lineHeight: 20 },
  segment: {
    flexDirection: "row",
    backgroundColor: colors.surface,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    paddingHorizontal: space.sm,
    paddingVertical: space.sm,
    gap: 6,
  },
  segBtn: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 5,
    paddingVertical: 9,
    borderRadius: radius.md,
  },
  segBtnActive: { backgroundColor: colors.surface2 },
  segText: { color: colors.faint, fontSize: 12, fontWeight: "700" },
  hint: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.lg,
    padding: space.md,
    marginBottom: space.md,
  },
  hintTitle: { color: colors.text, fontWeight: "800", fontSize: 15, marginBottom: 6 },
  hintText: { color: colors.muted, fontSize: 13, lineHeight: 19 },
  bubble: {
    borderRadius: radius.lg,
    padding: space.md,
    marginBottom: space.sm + 2,
    maxWidth: "92%",
  },
  userBubble: { backgroundColor: colors.accent, alignSelf: "flex-end" },
  aiBubble: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    alignSelf: "flex-start",
  },
  clarifyBubble: { borderColor: colors.accent2 },
  aiTag: { flexDirection: "row", alignItems: "center", gap: 6, marginBottom: 6 },
  brandDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: colors.accent },
  aiTagText: { color: colors.accent, fontSize: 12, fontWeight: "800" },
  bubbleText: { color: colors.text, fontSize: 14, lineHeight: 21 },
  thinking: { color: colors.muted, marginTop: 8, fontSize: 13 },
  clarifyBlock: {
    alignSelf: "flex-start",
    maxWidth: "92%",
    marginBottom: space.sm + 2,
    marginTop: -8,
    gap: space.sm,
  },
  clarifyQuestion: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.md,
    padding: space.sm + 2,
  },
  clarifyQuestionText: { color: colors.text, fontSize: 13, fontWeight: "700", marginBottom: 8 },
  clarifyOptions: { gap: 6 },
  clarifyOptBtn: {
    backgroundColor: colors.surface2,
    borderWidth: 1,
    borderColor: colors.accent2,
    borderRadius: radius.sm,
    paddingVertical: 9,
    paddingHorizontal: 12,
  },
  clarifyOptText: { color: colors.accent2, fontSize: 13, fontWeight: "700" },
  clarifyFreeHint: { color: colors.faint, fontSize: 11, fontStyle: "italic", marginLeft: 4 },
  clarifyNudge: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    marginBottom: 2,
  },
  clarifyNudgeText: { color: colors.accent2, fontSize: 11, fontWeight: "600" },
  inputBar: {
    padding: space.sm + 2,
    paddingBottom: space.md,
    borderTopWidth: 1,
    borderTopColor: colors.border,
    backgroundColor: colors.surface,
    gap: space.sm,
  },
  pillRow: { flexDirection: "row", alignItems: "center", gap: space.sm },
  modelPill: {
    flexDirection: "row",
    alignItems: "center",
    alignSelf: "flex-start",
    gap: 6,
    backgroundColor: colors.surface2,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.xl,
    paddingHorizontal: 12,
    paddingVertical: 6,
    maxWidth: "60%",
  },
  modelPillText: { color: colors.text, fontSize: 12, fontWeight: "700" },
  stopPill: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    backgroundColor: colors.surface2,
    borderWidth: 1,
    borderColor: colors.danger,
    borderRadius: radius.xl,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  stopPillText: { color: colors.danger, fontSize: 12, fontWeight: "700" },
  stopBtn: {
    width: 50,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.danger,
    alignItems: "center",
    justifyContent: "center",
  },
  modelPanel: {
    maxHeight: 300,
    backgroundColor: colors.surface2,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.md,
    padding: space.sm,
    marginBottom: space.sm,
  },
  modelPanelTitle: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: "800",
    marginBottom: space.sm,
    marginLeft: 4,
  },
  inputRow: { flexDirection: "row", alignItems: "flex-end", gap: space.sm },
  chatInput: {
    flex: 1,
    maxHeight: 130,
    backgroundColor: colors.surface2,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.lg,
    color: colors.text,
    paddingHorizontal: space.md,
    paddingVertical: 12,
    fontSize: 15,
  },
  sendBtn: {
    width: 46,
    height: 46,
    borderRadius: 23,
    backgroundColor: colors.accent,
    alignItems: "center",
    justifyContent: "center",
  },
  fileCard: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.md,
    marginBottom: space.sm,
    overflow: "hidden",
  },
  fileHead: { flexDirection: "row", alignItems: "center", gap: 8, padding: space.md },
  filePath: { flex: 1, color: colors.text, fontSize: 13, fontWeight: "700" },
  codeWrap: { backgroundColor: colors.code, padding: space.md, maxHeight: 320 },
  code: { color: "#B8E6C8", fontFamily: "monospace", fontSize: 12, lineHeight: 18 },
  infoCard: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.lg,
    padding: space.md,
    marginBottom: space.md,
  },
  infoTitle: { color: colors.text, fontWeight: "800", fontSize: 15, marginBottom: 6 },
  infoText: { color: colors.muted, fontSize: 13, lineHeight: 19 },
  warnText: { color: colors.warn, fontSize: 12, marginTop: 8, textAlign: "center" },
  sectionTitle: { color: colors.text, fontWeight: "800", fontSize: 15, marginBottom: space.sm },
  passCard: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.md,
    padding: space.md,
    marginBottom: space.sm,
  },
  passHead: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  passTitle: { color: colors.text, fontWeight: "800", flex: 1, marginRight: space.sm },
  badge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 20 },
  badgeText: { color: "#04140B", fontWeight: "800", fontSize: 11 },
  issue: { flexDirection: "row", gap: 8, marginTop: 10 },
  dot: { width: 8, height: 8, borderRadius: 4, marginTop: 5 },
  issueFile: { color: colors.accent2, fontSize: 12, fontWeight: "700" },
  issueDesc: { color: colors.text, fontSize: 13, marginTop: 2, lineHeight: 18 },
  issueFix: { color: colors.muted, fontSize: 12, marginTop: 3, fontStyle: "italic" },
  passSummary: { color: colors.muted, fontSize: 12, marginTop: 10 },
  label: { color: colors.muted, fontSize: 12, fontWeight: "700", marginBottom: 6, marginTop: 12 },
  input: {
    backgroundColor: colors.surface2,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.md,
    color: colors.text,
    paddingHorizontal: space.md,
    paddingVertical: 13,
    fontSize: 15,
  },
  commitRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingVertical: 6,
  },
  commitPath: { color: colors.text, fontSize: 13, flexShrink: 1 },
  commitErr: { color: colors.danger, fontSize: 11 },
  liveRow: { flexDirection: "row", alignItems: "center", gap: 8, marginTop: space.sm },
  confirmWrap: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.6)",
    alignItems: "center",
    justifyContent: "center",
    padding: space.lg,
  },
  confirmCard: {
    width: "100%",
    backgroundColor: colors.surface,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: space.lg,
    alignItems: "center",
    gap: 10,
  },
  confirmTitle: { color: colors.text, fontSize: 18, fontWeight: "800" },
  confirmText: { color: colors.muted, fontSize: 14, textAlign: "center", lineHeight: 20 },
  confirmRow: { flexDirection: "row", gap: space.sm, marginTop: space.sm, width: "100%" },
  confirmBtn: { flex: 1, height: 48, borderRadius: radius.md, alignItems: "center", justifyContent: "center" },
  cancelBtn: { backgroundColor: colors.surface2, borderWidth: 1, borderColor: colors.border },
  delBtn: { backgroundColor: colors.danger },
  cancelText: { color: colors.text, fontWeight: "800", fontSize: 15 },
  delText: { color: "#fff", fontWeight: "800", fontSize: 15 },
  modelWrap: { flex: 1, backgroundColor: "rgba(0,0,0,0.55)" },
  modelSheet: {
    backgroundColor: colors.surface,
    borderTopLeftRadius: radius.xl,
    borderTopRightRadius: radius.xl,
    padding: space.lg,
    paddingBottom: 40,
    borderTopWidth: 1,
    borderColor: colors.border,
  },
  mHandle: {
    width: 44,
    height: 5,
    borderRadius: 3,
    backgroundColor: colors.border,
    alignSelf: "center",
    marginBottom: space.md,
  },
  modelTitle: { color: colors.text, fontSize: 19, fontWeight: "800", marginBottom: space.sm },
  modelRow: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: colors.surface2,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.md,
    padding: space.md,
    marginBottom: space.sm,
  },
  modelRowActive: { borderColor: colors.accent },
  modelRowDisabled: { opacity: 0.4 },
  modelLabel: { color: colors.text, fontSize: 15, fontWeight: "700" },
  modelLabelDisabled: { color: colors.faint },
  modelHint: { color: colors.muted, fontSize: 12, marginTop: 2 },
});