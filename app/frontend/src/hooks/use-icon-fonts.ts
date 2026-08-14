import { useFonts } from "expo-font";
import Ionicons from "@expo/vector-icons/Ionicons";

// Loads the icon font used by @expo/vector-icons' Ionicons before any
// screen renders. This avoids Ionicons' internal componentDidMount fallback,
// which can throw on some Android environments if a font isn't registered
// yet when the first <Ionicons> mounts.
export function useIconFonts() {
  const [loaded, error] = useFonts({
    ...Ionicons.font,
  });

  return [loaded, error] as const;
}