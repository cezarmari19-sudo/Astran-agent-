import React, { useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  Pressable,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
} from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { colors, radius, space } from "@/src/theme";
import { PrimaryButton, useToast } from "@/src/components";
import { api } from "@/src/api";

type Mode = "login" | "register";

function isValidEmail(email: string) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim());
}

export default function AuthScreen() {
  const router = useRouter();
  const { show, Toast } = useToast();
  const [mode, setMode] = useState<Mode>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const isRegister = mode === "register";

  const validate = (): string | null => {
    if (!isValidEmail(email)) return "Introdu un email valid.";
    if (password.length < 8) return "Parola trebuie sa aiba cel putin 8 caractere.";
    if (isRegister && password !== confirmPassword) return "Parolele nu coincid.";
    return null;
  };

  const submit = async () => {
    const error = validate();
    if (error) {
      show(error, "err");
      return;
    }
    setBusy(true);
    try {
      if (isRegister) {
        await api.register(email.trim().toLowerCase(), password);
      } else {
        await api.login(email.trim().toLowerCase(), password);
      }
      router.replace("/(tabs)");
    } catch (e: any) {
      show(e.message, "err");
    } finally {
      setBusy(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={{ flex: 1, backgroundColor: colors.bg }}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        keyboardShouldPersistTaps="handled"
      >
        <View style={styles.brandBlock}>
          <View style={styles.brandDot} />
          <Text style={styles.brandTitle}>Astran</Text>
          <Text style={styles.brandSub}>
            {isRegister ? "Creeaza-ti contul" : "Autentifica-te pentru a continua"}
          </Text>
        </View>

        <View style={styles.form}>
          <Text style={styles.label}>Email</Text>
          <TextInput
            testID="auth-email-input"
            style={styles.input}
            placeholder="nume@exemplu.com"
            placeholderTextColor={colors.faint}
            autoCapitalize="none"
            autoCorrect={false}
            keyboardType="email-address"
            value={email}
            onChangeText={setEmail}
          />

          <Text style={styles.label}>Parola</Text>
          <View style={styles.passwordRow}>
            <TextInput
              testID="auth-password-input"
              style={[styles.input, { flex: 1 }]}
              placeholder="Minim 8 caractere"
              placeholderTextColor={colors.faint}
              autoCapitalize="none"
              secureTextEntry={!showPassword}
              value={password}
              onChangeText={setPassword}
            />
            <Pressable
              testID="toggle-password-visibility"
              onPress={() => setShowPassword((s) => !s)}
              style={styles.eyeBtn}
              hitSlop={10}
            >
              <Ionicons
                name={showPassword ? "eye-off-outline" : "eye-outline"}
                size={20}
                color={colors.faint}
              />
            </Pressable>
          </View>

          {isRegister && (
            <>
              <Text style={styles.label}>Confirma parola</Text>
              <TextInput
                testID="auth-confirm-password-input"
                style={styles.input}
                placeholder="Repeta parola"
                placeholderTextColor={colors.faint}
                autoCapitalize="none"
                secureTextEntry={!showPassword}
                value={confirmPassword}
                onChangeText={setConfirmPassword}
              />
            </>
          )}

          <View style={{ height: space.md }} />
          <PrimaryButton
            testID="auth-submit-btn"
            title={isRegister ? "Creeaza cont" : "Autentificare"}
            icon={isRegister ? "person-add" : "log-in"}
            loading={busy}
            onPress={submit}
          />

          <Pressable
            testID="auth-toggle-mode"
            onPress={() => {
              setMode(isRegister ? "login" : "register");
              setPassword("");
              setConfirmPassword("");
            }}
            style={styles.switchModeBtn}
          >
            <Text style={styles.switchModeText}>
              {isRegister
                ? "Ai deja cont? Autentifica-te"
                : "Nu ai cont? Creeaza unul"}
            </Text>
          </Pressable>
        </View>
      </ScrollView>
      {Toast}
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  scrollContent: {
    flexGrow: 1,
    justifyContent: "center",
    padding: space.lg,
  },
  brandBlock: {
    alignItems: "center",
    marginBottom: space.xl,
  },
  brandDot: {
    width: 14,
    height: 14,
    borderRadius: 7,
    backgroundColor: colors.accent,
    marginBottom: space.sm,
  },
  brandTitle: {
    color: colors.text,
    fontSize: 28,
    fontWeight: "800",
  },
  brandSub: {
    color: colors.muted,
    fontSize: 14,
    marginTop: 6,
  },
  form: {
    gap: 2,
  },
  label: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: "700",
    marginBottom: 6,
    marginTop: 14,
  },
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
  passwordRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: space.sm,
  },
  eyeBtn: {
    padding: 10,
  },
  switchModeBtn: {
    marginTop: space.lg,
    alignItems: "center",
    paddingVertical: space.sm,
  },
  switchModeText: {
    color: colors.accent2,
    fontSize: 13,
    fontWeight: "700",
  },
});
