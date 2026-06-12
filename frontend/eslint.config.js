import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import globals from "globals";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist"] },
  {
    files: ["**/*.{ts,tsx}"],
    extends: [
      js.configs.recommended,
      ...tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
    ],
    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser,
    },
    rules: {
      // This build doesn't run React Compiler, so its "compilation
      // skipped" diagnostics don't apply here.
      "react-hooks/preserve-manual-memoization": "off",
    },
  },
  {
    files: ["src/sw.ts"],
    languageOptions: {
      globals: globals.serviceworker,
    },
  },
);
