import React, { useEffect, useState } from "react";
import { View, Text, StyleSheet, ScrollView, TextInput } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { colors, radius, space } from "@/src/theme";
import { Header, PrimaryButton, useToast } from "@/src/components";
import { api } from "@/src/api";
import { storage } from "@/src/utils/storage";

export default function SettingsScreen() {
  const { show, Toast } = useToast();
  const [token, setToken] = useState("");
  const [checking, setChecking] = useState(false);
  const [ghInfo, setGhInfo] = useState<{ login: string; count: number } | null>(null);
  const [health, setHealth] = useState<any>(null);

  useEffect(() => {
    (async () => {
      setToken((await storage.secureGet("gh_token", "")) || "");
      try {
        setHealth(await api.health());
      } catch {}
    })();
  }, []);

  const verify = async () => {
    if (!token.trim()) return show("Adaugă un token", "err");
    setChecking(true);
    try {
      await storage.secureSet("gh_token", token.trim());
      const res = await api.githubRepos(token.trim());
      setGhInfo({ login: res.login, count: res.repos.length });
      show(`Conectat ca ${res.login}`);
    } catch (e: any) {
      setGhInfo(null);
      show(e.message, "err");
    } finally {
      setChecking(false);
    }
  };

  return (
    <View style={styles.container}>
      <Header title="Setări" subtitle="Cont GitHub, securitate & status" />
      <ScrollView contentContainerStyle={{ padding: space.md, paddingBottom: 120 }}>
        <View style={styles.section}>
          <View style={styles.rowHead}>
            <Ionicons name="logo-github" size={20} color={colors.text} />
            <Text style={styles.sectionTitle}>GitHub</Text>
          </View>
          <Text style={styles.help}>
            Creează un token în GitHub → Settings → Developer settings → Personal access
            tokens, cu permisiune „repo". Token-ul e salvat securizat pe telefon.
          </Text>
          <TextInput
            testID="settings-gh-token"
            style={styles.input}
            placeholder="ghp_..."
            placeholderTextColor={colors.faint}
            secureTextEntry
            autoCapitalize="none"
            value={token}
            onChangeText={setToken}
          />
          <View style={{ height: space.sm + 4 }} />
          <PrimaryButton
            testID="verify-gh-btn"
            title="Verifică & salvează token"
            icon="checkmark-circle"
            loading={checking}
            onPress={verify}
          />
          {ghInfo && (
            <View style={styles.ghCard}>
              <Ionicons name="checkmark-circle" size={18} color={colors.accent} />
              <Text style={styles.ghText}>
                {ghInfo.login} • {ghInfo.count} repository-uri
              </Text>
            </View>
          )}
        </View>

        <View style={styles.section}>
          <View style={styles.rowHead}>
            <Ionicons name="shield-checkmark" size={20} color={colors.accent} />
            <Text style={styles.sectionTitle}>Securitate cheie AI</Text>
          </View>
          <Text style={styles.help}>
            Cheia Gemini stă DOAR pe server (backend) și nu ajunge niciodată în aplicație.
            Frontend-ul vorbește doar cu API-ul tău, deci cheia nu poate fi furată din app.
          </Text>
          <View style={styles.statusRow}>
            <View
              style={[
                styles.statusDot,
                { backgroundColor: health?.ai_key_configured ? colors.accent : colors.danger },
              ]}
            />
            <Text style={styles.statusText}>
              {health?.ai_key_configured
                ? "Cheie AI configurată pe server ✓"
                : "Cheie AI neconfigurată"}
            </Text>
          </View>
        </View>

        <View style={styles.section}>
          <View style={styles.rowHead}>
            <Ionicons name="sparkles" size={20} color={colors.accent2} />
            <Text style={styles.sectionTitle}>Despre</Text>
          </View>
          <Text style={styles.help}>
            AI Builder — planifică și construiește aplicații cu Gemini 2.5 Flash. Un agent de
            verificare rulează în buclă până nu mai găsește probleme, apoi confirmă de 2 ori.
            Unelte: calculator, căutare web (SearXNG), notițe și commit direct pe GitHub.
          </Text>
        </View>
      </ScrollView>
      {Toast}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  section: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.lg,
    padding: space.md,
    marginBottom: space.md,
  },
  rowHead: { flexDirection: "row", alignItems: "center", gap: 8, marginBottom: 8 },
  sectionTitle: { color: colors.text, fontSize: 17, fontWeight: "800" },
  help: { color: colors.muted, fontSize: 13, lineHeight: 19, marginBottom: 12 },
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
  ghCard: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginTop: 12,
    backgroundColor: colors.surface2,
    borderRadius: radius.md,
    padding: space.md,
  },
  ghText: { color: colors.text, fontSize: 14, fontWeight: "700" },
  statusRow: { flexDirection: "row", alignItems: "center", gap: 8 },
  statusDot: { width: 10, height: 10, borderRadius: 5 },
  statusText: { color: colors.text, fontSize: 14 },
});