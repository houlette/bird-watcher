/**
 * Browser-side Web Push helpers. Drives the Settings page's subscribe UI
 * and registers the subscription with our backend so the server knows where
 * to send notifications.
 */

export type PushState =
  | { kind: "unsupported" }
  | { kind: "not_configured" }    // server returned no VAPID public key
  | { kind: "denied" }             // browser permission denied
  | { kind: "subscribed"; endpoint: string; notify_window_days: number }
  | { kind: "unsubscribed" };

export function isPushSupported(): boolean {
  return (
    typeof window !== "undefined" &&
    "serviceWorker" in navigator &&
    "PushManager" in window &&
    "Notification" in window
  );
}

async function fetchVapidPublicKey(): Promise<string | null> {
  const r = await fetch("/api/push/vapid_public_key");
  if (!r.ok) return null;
  const { public_key } = (await r.json()) as { public_key: string };
  return public_key || null;
}

function urlBase64ToUint8Array(base64: string): Uint8Array {
  const padding = "=".repeat((4 - (base64.length % 4)) % 4);
  const normalized = (base64 + padding).replace(/-/g, "+").replace(/_/g, "/");
  const raw = atob(normalized);
  const out = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
  return out;
}

export async function getCurrentSubscription(): Promise<PushSubscription | null> {
  if (!isPushSupported()) return null;
  const reg = await navigator.serviceWorker.ready;
  return reg.pushManager.getSubscription();
}

export async function getState(): Promise<PushState> {
  if (!isPushSupported()) return { kind: "unsupported" };
  if (Notification.permission === "denied") return { kind: "denied" };
  const sub = await getCurrentSubscription();
  if (sub) {
    // We can't query notify_window_days from the browser subscription itself.
    // The backend is authoritative; for UI display we just stash the last
    // requested window in localStorage.
    const stored = Number(localStorage.getItem("notify_window_days") ?? "30");
    return { kind: "subscribed", endpoint: sub.endpoint, notify_window_days: stored };
  }
  return { kind: "unsubscribed" };
}

export async function subscribe(notify_window_days: number): Promise<PushState> {
  if (!isPushSupported()) return { kind: "unsupported" };

  const publicKey = await fetchVapidPublicKey();
  if (!publicKey) return { kind: "not_configured" };

  const permission = await Notification.requestPermission();
  if (permission !== "granted") return { kind: "denied" };

  const reg = await navigator.serviceWorker.ready;
  let sub = await reg.pushManager.getSubscription();
  if (!sub) {
    sub = await reg.pushManager.subscribe({
      userVisibleOnly: true,
      // Cast: TS's newer lib.dom types declare BufferSource as needing an
      // ArrayBuffer-backed view specifically (not ArrayBufferLike). At
      // runtime our Uint8Array is always ArrayBuffer-backed; the cast tells
      // the type system so without forcing every caller to widen.
      applicationServerKey: urlBase64ToUint8Array(publicKey) as BufferSource,
    });
  }

  const json = sub.toJSON();
  const r = await fetch("/api/push/subscribe", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      endpoint: json.endpoint,
      keys: { p256dh: json.keys?.p256dh, auth: json.keys?.auth },
      notify_window_days,
    }),
  });
  if (!r.ok) throw new Error(`subscribe failed: ${r.status}`);
  localStorage.setItem("notify_window_days", String(notify_window_days));
  return { kind: "subscribed", endpoint: sub.endpoint, notify_window_days };
}

export async function unsubscribe(): Promise<PushState> {
  if (!isPushSupported()) return { kind: "unsupported" };
  const sub = await getCurrentSubscription();
  if (sub) {
    await fetch(`/api/push/subscribe?endpoint=${encodeURIComponent(sub.endpoint)}`, {
      method: "DELETE",
    });
    await sub.unsubscribe();
  }
  return { kind: "unsubscribed" };
}
