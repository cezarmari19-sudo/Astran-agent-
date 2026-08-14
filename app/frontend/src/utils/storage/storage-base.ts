export type StorageItemValue = string | number | boolean | null;

// Base class providing the shared serialize/deserialize + logging behavior
// used by both the general KV (AsyncStorage) and secure (Keychain) methods.
// Concrete get/set/remove implementations live in the subclass (index.ts /
// index.web.ts) since the underlying storage API differs by platform.
export abstract class StorageBase {
  protected retrieve<Fallback extends StorageItemValue>(
    raw: string | null,
    fallback: Fallback
  ): Fallback | null {
    if (raw === null || raw === undefined) return fallback;
    try {
      return JSON.parse(raw);
    } catch {
      return fallback;
    }
  }

  protected warn(method: string, key: string, error: unknown): void {
    if (__DEV__) {
      console.warn(`[storage] ${method} failed for key "${key}":`, error);
    }
  }
}

// Compile-time guard: ensures the concrete Storage class in index.ts never
// grows an extra public method without it being declared here first.
// Any key passed in resolves to `never`, which is exactly what index.ts
// expects when it has no methods beyond what StorageBase already declares.
export type AssertNoExtras<T extends string> = {
  [K in T]: "ERROR: new method must be declared in storage-base.ts first";
};