import React, { useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  TextInput,
  ScrollView,
  ActivityIndicator,
} from "react-native";
import * as WebBrowser from "expo-web-browser";
import { Ionicons } from "@expo/vector-icons";
import { colors, radius, space } from "@/src/theme";
import { Header, useToast } from "@/src/components";
import { api } from "@/src/api";

export default function ToolsScreen() {
  const { show, Toast } = useToast();
  const [mode, setMode] = useState<"calc" | "search">("calc");

  return (
    <View style={styles.container}>
      <Header title="Unelte" subtitle="Uneltele pe care le folosește și AI-ul" />
      <View style={styles.switch}>
        <Pressable
          testID="tool-calc"
          onPress={() => setMode("calc")}
          style={[styles.swBtn, mode === "calc" && styles.swActive]}
        >
          <Ionicons name="calculator" size={16} color={mode === "calc" ? colors.accent : colors.faint} />
          <Text style={[styles.swText, mode === "calc" && { color: colors.text }]}>Calculator</Text>
        </Pressable>
        <Pressable
          testID="tool-search"
          onPress={() => setMode("search")}
          style={[styles.swBtn, mode === "search" && styles.swActive]}
        >
          <Ionicons name="search" size={16} color={mode === "search" ? colors.accent : colors.faint} />
          <Text style={[styles.swText, mode === "search" && { color: colors.text }]}>Căutare web</Text>
        </Pressable>
      </View>
      {mode === "calc" ? <Calculator show={show} /> : <WebSearch show={show} />}
      {Toast}
    </View>
  );
}

function Calculator({ show }: any) {
  const [expr, setExpr] = useState("");
  const [result, setResult] = useState("");
  const keys = ["7", "8", "9", "/", "4", "5", "6", "*", "1", "2", "3", "-", "0", ".", "(", ")", "C", "="];

  const press = async (k: string) => {
    if (k === "C") {
      setExpr("");
      setResult("");
      return;
    }
    if (k === "=") {
      if (!expr) return;
      try {
        const res = await api.calculator(expr);
        setResult(String(res.result));
      } catch (e: any) {
        show(e.message, "err");
      }
      return;
    }
    setExpr((p) => p + k);
  };

  return (
    <View style={{ flex: 1, padding: space.md }}>
      <View style={styles.display}>
        <Text style={styles.exprText}>{expr || "0"}</Text>
        <Text style={styles.resultText}>{result ? `= ${result}` : ""}</Text>
      </View>
      <View style={styles.grid}>
        {keys.map((k) => (
          <Pressable
            key={k}
            testID={`calc-key-${k}`}
            onPress={() => press(k)}
            style={({ pressed }) => [
              styles.key,
              (k === "=" ) && styles.keyEq,
              (k === "C") && styles.keyClear,
              pressed && { opacity: 0.7 },
            ]}
          >
            <Text
              style={[
                styles.keyText,
                k === "=" && { color: "#04140B" },
                k === "C" && { color: colors.danger },
              ]}
            >
              {k}
            </Text>
          </Pressable>
        ))}
      </View>
    </View>
  );
}

function WebSearch({ show }: any) {
  const [q, setQ] = useState("");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any[]>([]);
  const [note, setNote] = useState("");

  const run = async () => {
    if (!q.trim()) return;
    setLoading(true);
    setNote("");
    try {
      const res = await api.websearch(q.trim());
      setResults(res.results || []);
      if (!res.results?.length) setNote(res.note || "Niciun rezultat.");
    } catch (e: any) {
      show(e.message, "err");
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={{ flex: 1 }}>
      <View style={styles.searchBar}>
        <TextInput
          testID="search-input"
          style={styles.searchInput}
          placeholder="Caută pe internet (SearXNG)…"
          placeholderTextColor={colors.faint}
          value={q}
          onChangeText={setQ}
          onSubmitEditing={run}
          returnKeyType="search"
        />
        <Pressable testID="search-go" onPress={run} style={styles.searchGo}>
          <Ionicons name="search" size={20} color="#04140B" />
        </Pressable>
      </View>
      {loading ? (
        <View style={styles.center}>
          <ActivityIndicator color={colors.accent} />
        </View>
      ) : (
        <ScrollView contentContainerStyle={{ padding: space.md, paddingBottom: 120 }}>
          {note ? <Text style={styles.note}>{note}</Text> : null}
          {results.map((r, i) => (
            <Pressable
              key={i}
              testID={`search-result-${i}`}
              onPress={() => r.url && WebBrowser.openBrowserAsync(r.url)}
              style={styles.resCard}
            >
              <Text style={styles.resTitle} numberOfLines={2}>
                {r.title}
              </Text>
              <Text style={styles.resUrl} numberOfLines={1}>
                {r.url}
              </Text>
              {r.content ? (
                <Text style={styles.resContent} numberOfLines={3}>
                  {r.content}
                </Text>
              ) : null}
            </Pressable>
          ))}
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  center: { flex: 1, alignItems: "center", justifyContent: "center" },
  switch: { flexDirection: "row", gap: space.sm, padding: space.md },
  swBtn: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    paddingVertical: 12,
    borderRadius: radius.md,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
  },
  swActive: { borderColor: colors.accent },
  swText: { color: colors.faint, fontWeight: "700", fontSize: 13 },
  display: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.lg,
    padding: space.lg,
    marginBottom: space.md,
    minHeight: 110,
    justifyContent: "flex-end",
  },
  exprText: { color: colors.text, fontSize: 28, fontWeight: "700", textAlign: "right" },
  resultText: { color: colors.accent, fontSize: 20, fontWeight: "800", textAlign: "right", marginTop: 6 },
  grid: { flexDirection: "row", flexWrap: "wrap", gap: space.sm },
  key: {
    width: "22.6%",
    aspectRatio: 1.35,
    backgroundColor: colors.surface2,
    borderRadius: radius.md,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
    borderColor: colors.border,
  },
  keyEq: { backgroundColor: colors.accent, borderColor: colors.accent },
  keyClear: { borderColor: colors.danger },
  keyText: { color: colors.text, fontSize: 22, fontWeight: "700" },
  searchBar: { flexDirection: "row", gap: space.sm, padding: space.md },
  searchInput: {
    flex: 1,
    backgroundColor: colors.surface2,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.md,
    color: colors.text,
    paddingHorizontal: space.md,
    paddingVertical: 13,
    fontSize: 15,
  },
  searchGo: {
    width: 50,
    borderRadius: radius.md,
    backgroundColor: colors.accent,
    alignItems: "center",
    justifyContent: "center",
  },
  note: { color: colors.muted, textAlign: "center", marginTop: 20, lineHeight: 20 },
  resCard: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.md,
    padding: space.md,
    marginBottom: space.sm,
  },
  resTitle: { color: colors.accent2, fontSize: 15, fontWeight: "700" },
  resUrl: { color: colors.faint, fontSize: 11, marginTop: 2 },
  resContent: { color: colors.muted, fontSize: 13, marginTop: 6, lineHeight: 18 },
});