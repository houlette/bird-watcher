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
  const [muteResidents, setMuteResidents] = useState(true);
  const [notifyDailyFirst, setNotifyDailyFirst] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getState().then((s) => {
      setState(s);
      if (s.kind === "subscribed") {
        setWindowDays(s.notify_window_days);
        setMuteResidents(s.mute_residents);
        setNotifyDailyFirst(s.notify_daily_first);
      }
    });
  }, []);

  const onSubscribe = async () => {
    setBusy(true);
    setError(null);
    try {
      const next = await subscribe(windowDays, muteResidents, notifyDailyFirst);
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
          Smart notification tiers for rare arrivals and daily visits without alert fatigue.
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
            <div className="fg-card p-4 space-y-4">
              <div>
                <label className="flex items-center justify-between cursor-pointer">
                  <span className="text-sm font-semibold text-ink">
                    Mute common resident birds
                  </span>
                  <input
                    type="checkbox"
                    checked={muteResidents}
                    onChange={(e) => setMuteResidents(e.target.checked)}
                    disabled={busy}
                    className="h-4 w-4 rounded border-line accent-[var(--accent)] cursor-pointer"
                  />
                </label>
                <p className="text-xs text-muted mt-1">
                  Silences real-time alerts for Mourning Doves, Rock Pigeons, and House Sparrows to eliminate notification fatigue.
                </p>
              </div>

              <div className="border-t border-line/60 pt-4">
                <label className="flex items-center justify-between cursor-pointer">
                  <span className="text-sm font-semibold text-ink">
                    Notify on first arrival of the day
                  </span>
                  <input
                    type="checkbox"
                    checked={notifyDailyFirst}
                    onChange={(e) => setNotifyDailyFirst(e.target.checked)}
                    disabled={busy}
                    className="h-4 w-4 rounded border-line accent-[var(--accent)] cursor-pointer"
                  />
                </label>
                <p className="text-xs text-muted mt-1">
                  Sends an alert the first time each feeder regular (Cardinals, Blue Jays, Woodpeckers) arrives each morning, then quiets down for subsequent visits.
                </p>
              </div>

              <div className="border-t border-line/60 pt-4">
                <label className="block text-sm font-semibold text-ink">
                  Rarity threshold: first sighting within{" "}
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
                <p className="text-xs text-muted mt-2">
                  A bird species that hasn't been seen in this many days triggers a rare visitor alert.
                </p>
              </div>
            </div>

            {state.kind === "subscribed" ? (
              <div className="flex flex-wrap gap-2.5">
                <button onClick={onSubscribe} disabled={busy} className="fg-btn-primary px-4 py-2 text-sm">
                  {busy ? "Updating…" : "Save notification preferences"}
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
