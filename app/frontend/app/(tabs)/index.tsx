import React, { useCallback, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  Pressable,
  TextInput,
  Modal,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
} from "react-native";
import { useRouter, useFocusEffect } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { colors, radius, space } from "@/src/theme";
import { Header, PrimaryButton, useToast } from "@/src/components";
import { api, Project } from "@/src/api";

export default function BuildScreen() {
  const router = useRouter();
  const { show, Toast } = useToast();
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [modal, setModal] = useState(false);
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");
  const [creating, setCreating] = useState(false);
  const [confirmDel, setConfirmDel] = useState<Project | null>(null);
  const [deleting, setDeleting] = useState(false);

  const load = useCallback(async () => {
    try {
      setProjects(await api.listProjects());
    } catch (e: any) {
      show(e.message, "err");
    } finally {
      setLoading(false);
    }
  }, [show]);

  useFocusEffect(
    useCallback(() => {
      load();
    }, [load])
  );

  const create = async () => {
    if (!name.trim()) return show("Dă un nume proiectului", "err");
    setCreating(true);
    try {
      const p = await api.createProject(name.trim(), desc.trim());
      setModal(false);
      setName("");
      setDesc("");
      router.push(`/project/${p.id}`);
    } catch (e: any) {
      show(e.message, "err");
    } finally {
      setCreating(false);
    }
  };

  const renderItem = ({ item }: { item: Project }) => (
    <Pressable
      testID={`project-card-${item.id}`}
      onPress={() => router.push(`/project/${item.id}`)}
      style={({ pressed }) => [styles.card, pressed && { opacity: 0.75 }]}
    >
      <View style={styles.cardIcon}>
        <Ionicons name="cube" size={22} color={colors.accent} />
      </View>
      <View style={{ flex: 1 }}>
        <Text style={styles.cardTitle} numberOfLines={1}>
          {item.name}
        </Text>
        <Text style={styles.cardDesc} numberOfLines={2}>
          {item.description || "Fără descriere"}
        </Text>
        <Text style={styles.cardMeta}>
          {item.files?.length || 0} fișiere generate
        </Text>
      </View>
      <Ionicons name="chevron-forward" size={20} color={colors.faint} />
    </Pressable>
  );

  return (
    <View style={styles.container}>
      <Header title="AI Builder" subtitle="Planifică și construiește aplicații cu AI" />

      {loading ? (
        <View style={styles.center}>
          <ActivityIndicator color={colors.accent} />
        </View>
      ) : (
        <FlatList
          data={projects}
          keyExtractor={(i) => i.id}
          renderItem={renderItem}
          contentContainerStyle={{ padding: space.md, paddingBottom: 140 }}
          ListEmptyComponent={
            <View style={styles.empty}>
              <Ionicons name="rocket-outline" size={54} color={colors.faint} />
              <Text style={styles.emptyTitle}>Niciun proiect încă</Text>
              <Text style={styles.emptyText}>
                Creează un proiect și descrie ce aplicație vrei. Aria o planifică și
                scrie codul, apoi un agent verifică totul.
              </Text>
            </View>
          }
        />
      )}

      <Pressable
        testID="new-project-fab"
        onPress={() => setModal(true)}
        style={({ pressed }) => [styles.fab, pressed && { opacity: 0.85 }]}
      >
        <Ionicons name="add" size={26} color="#04140B" />
        <Text style={styles.fabText}>Proiect nou</Text>
      </Pressable>

      <Modal visible={modal} transparent animationType="slide" onRequestClose={() => setModal(false)}>
        <KeyboardAvoidingView
          behavior={Platform.OS === "ios" ? "padding" : "height"}
          style={styles.modalWrap}
        >
          <Pressable style={{ flex: 1 }} onPress={() => setModal(false)} />
          <View style={styles.sheet}>
            <View style={styles.sheetHandle} />
            <Text style={styles.sheetTitle}>Proiect nou</Text>
            <TextInput
              testID="project-name-input"
              placeholder="Nume (ex: Aplicație de fitness)"
              placeholderTextColor={colors.faint}
              style={styles.input}
              value={name}
              onChangeText={setName}
            />
            <TextInput
              testID="project-desc-input"
              placeholder="Descriere scurtă (opțional)"
              placeholderTextColor={colors.faint}
              style={[styles.input, { height: 90, textAlignVertical: "top" }]}
              multiline
              value={desc}
              onChangeText={setDesc}
            />
            <PrimaryButton
              testID="create-project-btn"
              title="Creează și deschide"
              icon="arrow-forward"
              loading={creating}
              onPress={create}
            />
          </View>
        </KeyboardAvoidingView>
      </Modal>

      <Modal
        visible={!!confirmDel}
        transparent
        animationType="fade"
        onRequestClose={() => setConfirmDel(null)}
      >
        <View style={styles.confirmWrap}>
          <View style={styles.confirmCard}>
            <Ionicons name="trash" size={30} color={colors.danger} />
            <Text style={styles.confirmTitle}>Ștergi acest chat?</Text>
            <Text style={styles.confirmText}>
              „{confirmDel?.name}" și tot codul + conversația se șterg definitiv.
            </Text>
            <View style={styles.confirmRow}>
              <Pressable
                testID="cancel-delete-btn"
                style={[styles.confirmBtn, styles.cancelBtn]}
                onPress={() => setConfirmDel(null)}
              >
                <Text style={styles.cancelText}>Anulează</Text>
              </Pressable>
              <Pressable
                testID="confirm-delete-btn"
                style={[styles.confirmBtn, styles.delBtn]}
                onPress={remove}
              >
                <Text style={styles.delText}>{deleting ? "Se șterge…" : "Șterge"}</Text>
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>
      {Toast}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  center: { flex: 1, alignItems: "center", justifyContent: "center" },
  card: {
    flexDirection: "row",
    alignItems: "center",
    gap: space.md,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.lg,
    padding: space.md,
    marginBottom: space.sm + 4,
  },
  cardIcon: {
    width: 46,
    height: 46,
    borderRadius: radius.md,
    backgroundColor: colors.surface2,
    alignItems: "center",
    justifyContent: "center",
  },
  cardTitle: { color: colors.text, fontSize: 16, fontWeight: "800" },
  cardDesc: { color: colors.muted, fontSize: 13, marginTop: 2 },
  cardMeta: { color: colors.accent, fontSize: 11, marginTop: 6, fontWeight: "700" },
  empty: { alignItems: "center", paddingTop: 90, paddingHorizontal: space.lg, gap: 10 },
  emptyTitle: { color: colors.text, fontSize: 18, fontWeight: "800", marginTop: 6 },
  emptyText: { color: colors.muted, fontSize: 14, textAlign: "center", lineHeight: 20 },
  fab: {
    position: "absolute",
    bottom: 104,
    right: space.md,
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    backgroundColor: colors.accent,
    paddingHorizontal: 18,
    height: 52,
    borderRadius: radius.xl,
  },
  fabText: { color: "#04140B", fontWeight: "800", fontSize: 15 },
  modalWrap: { flex: 1, backgroundColor: "rgba(0,0,0,0.55)" },
  sheet: {
    backgroundColor: colors.surface,
    borderTopLeftRadius: radius.xl,
    borderTopRightRadius: radius.xl,
    padding: space.lg,
    paddingBottom: 40,
    borderTopWidth: 1,
    borderColor: colors.border,
    gap: space.sm + 4,
  },
  sheetHandle: {
    width: 44,
    height: 5,
    borderRadius: 3,
    backgroundColor: colors.border,
    alignSelf: "center",
    marginBottom: space.sm,
  },
  sheetTitle: { color: colors.text, fontSize: 20, fontWeight: "800", marginBottom: 4 },
  input: {
    backgroundColor: colors.surface2,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.md,
    color: colors.text,
    paddingHorizontal: space.md,
    paddingVertical: 14,
    fontSize: 15,
  },
});