import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";

/**
 * Minimal single-slot snackbar. A new toast replaces any visible one
 * (the user only ever cares about the most recent action), auto-dismisses
 * after a few seconds, and can carry one action button — used for the
 * "Undo" affordance after a fast review tap.
 */
type ToastAction = { label: string; onAction: () => void };
type ToastState = { id: number; message: string; action?: ToastAction };
type ShowToast = (message: string, action?: ToastAction) => void;

const ToastCtx = createContext<ShowToast>(() => {});

// Actions (Undo) get a longer window than plain confirmations — you need a
// beat to realize you mis-tapped.
const PLAIN_MS = 2800;
const ACTION_MS = 6000;

export function useToast(): ShowToast {
  return useContext(ToastCtx);
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toast, setToast] = useState<ToastState | null>(null);
  const timer = useRef<number | null>(null);

  const clearTimer = () => {
    if (timer.current !== null) {
      clearTimeout(timer.current);
      timer.current = null;
    }
  };

  const dismiss = useCallback(() => {
    clearTimer();
    setToast(null);
  }, []);

  const show = useCallback<ShowToast>((message, action) => {
    clearTimer();
    setToast({ id: Date.now(), message, action });
    timer.current = window.setTimeout(
      () => setToast(null),
      action ? ACTION_MS : PLAIN_MS,
    );
  }, []);

  useEffect(() => clearTimer, []);

  return (
    <ToastCtx.Provider value={show}>
      {children}
      {toast && (
        <div
          key={toast.id}
          className="fixed left-1/2 -translate-x-1/2 z-[60] bottom-[calc(5.5rem+env(safe-area-inset-bottom,0px))] flex items-center gap-3 rounded-full border border-line bg-surface px-4 py-2.5 text-sm text-ink shadow-pop max-w-[calc(100vw-2rem)]"
          role="status"
          aria-live="polite"
        >
          <span className="truncate">{toast.message}</span>
          {toast.action && (
            <button
              className="shrink-0 font-semibold text-leaf hover:underline underline-offset-2"
              onClick={() => {
                toast.action!.onAction();
                dismiss();
              }}
            >
              {toast.action.label}
            </button>
          )}
        </div>
      )}
    </ToastCtx.Provider>
  );
}
