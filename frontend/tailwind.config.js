/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        forest: "#2f5d50",
        cream: "#f7f4ed",
      },
    },
  },
  plugins: [],
};
