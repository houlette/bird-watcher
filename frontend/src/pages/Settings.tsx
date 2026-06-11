import { useEffect, useState } from "react";

import { getState, subscribe, unsubscribe, type PushState } from "../lib/push";

// Small note box for the non-actionable push states (unsupported / not
// configured / denied). Tinted with the rust accent so it reads as
// "attention needed" within the field-guide palette.
function NoteBox({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-card border border-[color-mix(in_oklab,var(--rust)_35%,var(--line))] bg-[color-mix(in_oklab,var(--rust)_8%,var(--card))] px-3.5 py-3 text-sm text-ink">
      {children}
    </div>
  );
}

function Code({ children }: { children: React.ReactNode }) {
  return (
    <code className="px-1 mx-0.5 rounded bg-panel border border-line text-[0.85em] text-ink">
      {children}
    </code>
  );
}

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

  if (!state) return <p className="text-muted mt-4">Loading…</p>;

  return (
    <div className="max-w-xl">
      <div className="mb-5">
        <div className="fg-overline">Preferences</div>
        <h2 className="font-serif font-medium text-2xl text-ink leading-tight mt-0.5">
          Notifications
        </h2>
        <p className="text-sm text-muted mt-1">
          A push alert when a rarely-seen species shows up at the feeder.
        </p>
      </div>

      <div className="space-y-5">
        {state.kind === "unsupported" && (
          <NoteBox>
            This browser doesn't support Web Push. Try Chrome on Android, or any recent
            desktop browser.
          </NoteBox>
        )}

        {state.kind === "not_configured" && (
          <NoteBox>
            Push notifications aren't configured on the server yet. Run{" "}
            <Code>scripts/generate_vapid_keys.py</Code> and restart the backend.
          </NoteBox>
        )}

        {state.kind === "denied" && (
          <NoteBox>
            Notification permission is blocked in your browser settings. Open site
            permissions and re-allow notifications, then reload this page.
          </NoteBox>
        )}

        {(state.kind === "subscribed" || state.kind === "unsubscribed") && (
          <>
            <div className="fg-card p-4">
              <label className="block text-sm font-semibold text-ink">
                Notify on first sighting within the last{" "}
                <span className="font-serif text-leaf text-lg tnum">{windowDays}</span> days
              </label>
              <input
                type="range"
                min={1}
                max={90}
                value={windowDays}
                onChange={(e) => setWindowDays(Number(e.target.value))}
                className="fg-range w-full mt-3"
                disabled={busy}
              />
              <div className="flex justify-between text-[11px] text-faint mt-1 tnum">
                <span>1 day · chatty</span>
                <span>90 days · only memorable arrivals</span>
              </div>
              <p className="text-xs text-muted mt-3">
                A bird species that hasn't been seen in this many days triggers a push.
                Larger = quieter; smaller = chattier.
              </p>
            </div>

            {state.kind === "subscribed" ? (
              <div className="flex flex-wrap gap-2.5">
                <button onClick={onSubscribe} disabled={busy} className="fg-btn-primary px-4 py-2 text-sm">
                  {busy ? "Updating…" : "Save window setting"}
                </button>
                <button onClick={onUnsubscribe} disabled={busy} className="fg-btn-ghost px-4 py-2 text-sm">
                  Turn off notifications
                </button>
              </div>
            ) : (
              <button onClick={onSubscribe} disabled={busy} className="fg-btn-primary px-4 py-2 text-sm">
                {busy ? "Subscribing…" : "Enable bird notifications"}
              </button>
            )}
          </>
        )}

        {error && <p className="text-sm text-rust">{error}</p>}

        {state.kind === "subscribed" && (
          <p className="text-xs text-faint break-all">
            Subscribed endpoint: <Code>{state.endpoint.slice(0, 60)}…</Code>
          </p>
        )}
      </div>
    </div>
  );
}
