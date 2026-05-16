import { useEffect, useState } from "react";

import { getState, subscribe, unsubscribe, type PushState } from "../lib/push";

export default function Settings() {
  const [state, setState] = useState<PushState | null>(null);
  const [windowDays, setWindowDays] = useState(30);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getState().then((s) => {
      setState(s);
      if (s.kind === "subscribed") setWindowDays(s.notify_window_days);
    });
  }, []);

  const onSubscribe = async () => {
    setBusy(true);
    setError(null);
    try {
      const next = await subscribe(windowDays);
      setState(next);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const onUnsubscribe = async () => {
    setBusy(true);
    setError(null);
    try {
      setState(await unsubscribe());
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  if (!state) return <p className="text-slate-500">Loading…</p>;

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold">Notifications</h2>

      {state.kind === "unsupported" && (
        <p className="text-sm text-slate-600">
          This browser doesn't support Web Push. Try Chrome on Android, or any recent desktop browser.
        </p>
      )}

      {state.kind === "not_configured" && (
        <p className="text-sm text-amber-700">
          Push notifications aren't configured on the server yet. Run
          <code className="px-1 mx-1 bg-slate-100 rounded text-xs">
            scripts/generate_vapid_keys.py
          </code>
          and restart the backend.
        </p>
      )}

      {state.kind === "denied" && (
        <p className="text-sm text-amber-700">
          Notification permission is blocked in your browser settings. Open site permissions and re-allow notifications,
          then reload this page.
        </p>
      )}

      {(state.kind === "subscribed" || state.kind === "unsubscribed") && (
        <>
          <div className="space-y-2">
            <label className="block text-sm font-medium">
              Notify on first sighting within the last
              <span className="ml-2 inline-block w-16 text-right">{windowDays}</span> days
            </label>
            <input
              type="range"
              min={1}
              max={90}
              value={windowDays}
              onChange={(e) => setWindowDays(Number(e.target.value))}
              className="w-full"
              disabled={busy}
            />
            <p className="text-xs text-slate-500">
              A bird species that hasn't been seen in this many days triggers a push. Larger = quieter (more
              memorable arrivals only). Smaller = chattier.
            </p>
          </div>

          {state.kind === "subscribed" ? (
            <div className="flex gap-2">
              <button
                onClick={onSubscribe}
                disabled={busy}
                className="px-4 py-2 bg-forest text-cream rounded text-sm font-medium disabled:opacity-50"
              >
                {busy ? "Updating…" : "Save window setting"}
              </button>
              <button
                onClick={onUnsubscribe}
                disabled={busy}
                className="px-4 py-2 bg-white border border-slate-300 rounded text-sm disabled:opacity-50"
              >
                Turn off notifications
              </button>
            </div>
          ) : (
            <button
              onClick={onSubscribe}
              disabled={busy}
              className="px-4 py-2 bg-forest text-cream rounded text-sm font-medium disabled:opacity-50"
            >
              {busy ? "Subscribing…" : "Enable bird notifications"}
            </button>
          )}
        </>
      )}

      {error && <p className="text-sm text-red-600">{error}</p>}

      {state.kind === "subscribed" && (
        <p className="text-xs text-slate-500 break-all">
          Subscribed endpoint: <code>{state.endpoint.slice(0, 60)}…</code>
        </p>
      )}
    </div>
  );
}
