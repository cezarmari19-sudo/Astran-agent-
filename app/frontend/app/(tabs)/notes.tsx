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
} from "react-native";
import { useFocusEffect } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { colors, radius, space } from "@/src/theme";
import { Header, PrimaryButton, useToast } from "@/src/components";
import { api } from "@/src/api";

type Note = { id: string; title: string; content: string; created_at: string };

export default function NotesScreen() {
  const { show, Toast } = useToast();
  const [notes, setNotes] = useState<Note[]>([]);
  const [modal, setModal] = useState(false);
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [saving, setSaving] = useState(false);

  const load = useCallback(async () => {
    try {
      setNotes(await api.listNotes());
    } catch (e: any) {
      show(e.message, "err");
    }
  }, [show]);

  useFocusEffect(
    useCallback(() => {
      load();
    }, [load])
  );

  const save = async () => {
    if (!title.trim()) return show("Adaugă un titlu", "err");
    setSaving(true);
    try {
      await api.createNote(title.trim(), content.trim());
      setModal(false);
      setTitle("");
      setContent("");
      load();
      show("Notiță salvată");
    } catch (e: any) {
      show(e.message, "err");
    } finally {
      setSaving(false);
    }
  };

  const remove = async (id: string) => {
    try {
      await api.deleteNote(id);
      setNotes((n) => n.filter((x) => x.id !== id));
    } catch (e: any) {
      show(e.message, "err");
    }
  };

  return (
    <View style={styles.container}>
      <Header title="Notițe" subtitle="Informații pe care le ține minte AI-ul" />
      <FlatList
        data={notes}
        keyExtractor={(i) => i.id}
        contentContainerStyle={{ padding: space.md, paddingBottom: 140 }}
        renderItem={({ item }) => (
          <View testID={`note-${item.id}`} style={styles.card}>
            <View style={{ flex: 1 }}>
              <Text style={styles.title}>{item.title}</Text>
              {item.content ? <Text style={styles.content}>{item.content}</Text> : null}
            </View>
            <Pressable testID={`delete-note-${item.id}`} onPress={() => remove(item.id)} hitSlop={10}>
              <Ionicons name="trash-outline" size={20} color={colors.danger} />
            </Pressable>
          </View>
        )}
        ListEmptyComponent={
          <View style={styles.empty}>
            <Ionicons name="document-text-outline" size={50} color={colors.faint} />
            <Text style={styles.emptyText}>Nicio notiță. Adaugă informații de reținut.</Text>
          </View>
        }
      />
      <Pressable testID="add-note-fab" onPress={() => setModal(true)} style={styles.fab}>
        <Ionicons name="add" size={26} color="#04140B" />
      </Pressable>

      <Modal visible={modal} transparent animationType="slide" onRequestClose={() => setModal(false)}>
        <KeyboardAvoidingView
          behavior={Platform.OS === "ios" ? "padding" : "height"}
          style={styles.modalWrap}
        >
          <Pressable style={{ flex: 1 }} onPress={() => setModal(false)} />
          <View style={styles.sheet}>
            <View style={styles.handle} />
            <Text style={styles.sheetTitle}>Notiță nouă</Text>
            <TextInput
              testID="note-title-input"
              placeholder="Titlu"
              placeholderTextColor={colors.faint}
              style={styles.input}
              value={title}
              onChangeText={setTitle}
            />
            <TextInput
              testID="note-content-input"
              placeholder="Conținut…"
              placeholderTextColor={colors.faint}
              style={[styles.input, { height: 120, textAlignVertical: "top" }]}
              multiline
              value={content}
              onChangeText={setContent}
            />
            <PrimaryButton testID="save-note-btn" title="Salvează" icon="save" loading={saving} onPress={save} />
          </View>
        </KeyboardAvoidingView>
      </Modal>
      {Toast}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  card: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: space.md,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.lg,
    padding: space.md,
    marginBottom: space.sm + 2,
  },
  title: { color: colors.text, fontSize: 16, fontWeight: "800" },
  content: { color: colors.muted, fontSize: 14, marginTop: 4, lineHeight: 20 },
  empty: { alignItems: "center", paddingTop: 90, gap: 12 },
  emptyText: { color: colors.muted, textAlign: "center", paddingHorizontal: space.lg },
  fab: {
    position: "absolute",
    bottom: 104,
    right: space.md,
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: colors.accent,
    alignItems: "center",
    justifyContent: "center",
  },
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
  handle: { width: 44, height: 5, borderRadius: 3, backgroundColor: colors.border, alignSelf: "center", marginBottom: space.sm },
  sheetTitle: { color: colors.text, fontSize: 20, fontWeight: "800" },
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