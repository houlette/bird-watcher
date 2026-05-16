/// <reference lib="webworker" />
/**
 * Service worker for BirdWatcher.
 *
 * Two jobs:
 *   1. Precache static PWA assets (handled by vite-plugin-pwa's injectManifest).
 *   2. Receive Web Push messages from the backend and surface them as system
 *      notifications. Click handling navigates the open PWA window to the
 *      detection, or opens a new one if the PWA isn't running.
 */
import { precacheAndRoute } from "workbox-precaching";

declare const self: ServiceWorkerGlobalScope;

// Precache manifest is injected at build time by vite-plugin-pwa.
precacheAndRoute(self.__WB_MANIFEST);

type PushPayload = {
  title?: string;
  body?: string;
  icon?: string;
  data?: {
    detection_id?: number;
    visit_id?: number;
    species?: string;
    url?: string;
  };
};

self.addEventListener("push", (event) => {
  if (!event.data) return;
  let payload: PushPayload;
  try {
    payload = event.data.json() as PushPayload;
  } catch {
    payload = { title: "BirdWatcher", body: event.data.text() };
  }
  event.waitUntil(
    self.registration.showNotification(payload.title ?? "BirdWatcher", {
      body: payload.body ?? "",
      icon: payload.icon ?? "/icons/icon-192.png",
      badge: "/icons/icon-192.png",
      data: payload.data ?? {},
      tag: payload.data?.species, // collapse repeats of the same species
    }),
  );
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const url = (event.notification.data?.url as string | undefined) ?? "/";
  event.waitUntil(
    self.clients.matchAll({ type: "window", includeUncontrolled: true }).then((clients) => {
      for (const client of clients) {
        if ("focus" in client) {
          // Navigate the existing window if it's a same-origin client.
          const wc = client as WindowClient;
          wc.navigate(url).catch(() => {});
          return wc.focus();
        }
      }
      return self.clients.openWindow(url);
    }),
  );
});

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));
