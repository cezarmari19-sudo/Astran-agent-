import React, { useEffect, useRef, useState, useCallback } from "react";
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  Animated,
  ActivityIndicator,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import { colors, radius, space } from "./theme";

export function Header({
  title,
  subtitle,
  onBack,
  right,
}: {
  title: string;
  subtitle?: string;
  onBack?: () => void;
  right?: React.ReactNode;
}) {
  const insets = useSafeAreaInsets();
  return (
    <View
      testID="app-header"
      style={[styles.header, { paddingTop: insets.top + space.sm }]}
    >
      <View style={styles.headerRow}>
        {onBack ? (
          <Pressable testID="header-back" onPress={onBack} hitSlop={12} style={styles.backBtn}>
            <Ionicons name="chevron-back" size={24} color={colors.text} />
          </Pressable>
        ) : (
          <View style={styles.brandDot} />
        )}
        <View style={{ flex: 1 }}>
          <Text style={styles.headerTitle} numberOfLines={1}>
            {title}
          </Text>
          {subtitle ? (
            <Text style={styles.headerSub} numberOfLines={1}>
              {subtitle}
            </Text>
          ) : null}
        </View>
        {right}
      </View>
    </View>
  );
}

export function PrimaryButton({
  title,
  onPress,
  loading,
  icon,
  testID,
  variant = "solid",
  disabled,
}: {
  title: string;
  onPress: () => void;
  loading?: boolean;
  icon?: any;
  testID?: string;
  variant?: "solid" | "ghost";
  disabled?: boolean;
}) {
  const ghost = variant === "ghost";
  return (
    <Pressable
      testID={testID}
      onPress={onPress}
      disabled={loading || disabled}
      style={({ pressed }) => [
        styles.btn,
        ghost ? styles.btnGhost : styles.btnSolid,
        (pressed || disabled) && { opacity: 0.7 },
      ]}
    >
      {loading ? (
        <ActivityIndicator color={ghost ? colors.text : "#04140B"} />
      ) : (
        <>
          {icon ? (
            <Ionicons
              name={icon}
              size={18}
              color={ghost ? colors.text : "#04140B"}
              style={{ marginRight: 8 }}
            />
          ) : null}
          <Text style={[styles.btnText, ghost && { color: colors.text }]}>{title}</Text>
        </>
      )}
    </Pressable>
  );
}

export function useToast() {
  const [msg, setMsg] = useState("");
  const [type, setType] = useState<"ok" | "err">("ok");
  const opacity = useRef(new Animated.Value(0)).current;
  const timer = useRef<any>(null);

  const show = useCallback(
    (m: string, t: "ok" | "err" = "ok") => {
      setMsg(m);
      setType(t);
      Animated.timing(opacity, { toValue: 1, duration: 180, useNativeDriver: true }).start();
      if (timer.current) clearTimeout(timer.current);
      timer.current = setTimeout(() => {
        Animated.timing(opacity, { toValue: 0, duration: 250, useNativeDriver: true }).start();
      }, 2800);
    },
    [opacity]
  );

  useEffect(() => () => timer.current && clearTimeout(timer.current), []);

  const Toast = (
    <Animated.View
      pointerEvents="none"
      testID="toast"
      style={[
        styles.toast,
        { opacity, borderColor: type === "ok" ? colors.accent : colors.danger },
      ]}
    >
      <Ionicons
        name={type === "ok" ? "checkmark-circle" : "alert-circle"}
        size={18}
        color={type === "ok" ? colors.accent : colors.danger}
      />
      <Text style={styles.toastText}>{msg}</Text>
    </Animated.View>
  );

  return { show, Toast };
}

const styles = StyleSheet.create({
  header: {
    backgroundColor: colors.bg,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
    paddingHorizontal: space.md,
    paddingBottom: space.md,
  },
  headerRow: { flexDirection: "row", alignItems: "center", gap: space.sm },
  backBtn: { padding: 2 },
  brandDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: colors.accent,
    marginRight: 2,
  },
  headerTitle: { color: colors.text, fontSize: 20, fontWeight: "800", letterSpacing: 0.2 },
  headerSub: { color: colors.muted, fontSize: 12, marginTop: 2 },
  btn: {
    height: 50,
    borderRadius: radius.md,
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    paddingHorizontal: space.md,
  },
  btnSolid: { backgroundColor: colors.accent },
  btnGhost: { backgroundColor: colors.surface2, borderWidth: 1, borderColor: colors.border },
  btnText: { color: "#04140B", fontSize: 16, fontWeight: "800" },
  toast: {
    position: "absolute",
    bottom: 96,
    alignSelf: "center",
    maxWidth: "90%",
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    backgroundColor: colors.surface2,
    borderWidth: 1,
    borderRadius: radius.md,
    paddingHorizontal: space.md,
    paddingVertical: 12,
  },
  toastText: { color: colors.text, fontSize: 13, flexShrink: 1 },
});