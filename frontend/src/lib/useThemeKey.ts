import { useEffect, useState } from "react";

/**
 * A string that changes whenever the Sage/Twilight class flips on <html>.
 *
 * Canvas surfaces read their palette from the CSS variables once and cache
 * it; passing this key as a prop is how they know to read it again. Shared
 * by the Art page and the tavern, which both draw with the theme's colours.
 */
export function useThemeKey(): string {
  const [key, setKey] = useState(() =>
    typeof document === "undefined" ? "" : document.documentElement.className
  );
  useEffect(() => {
    const ob = new MutationObserver(() => setKey(document.documentElement.className));
    ob.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => ob.disconnect();
  }, []);
  return key;
}
