/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0A0E13",
        panel: "#121821",
        panel2: "#171F29",
        hairline: "#232C36",
        text: "#E7ECF0",
        muted: "#8B98A5",
        verified: "#34D6A6",
        flagged: "#E8B04B",
        invalid: "#E5484D",
        stamp: "#5468FF",
      },
      fontFamily: {
        display: ["'Space Grotesk'", "sans-serif"],
        body: ["'IBM Plex Sans'", "sans-serif"],
        mono: ["'IBM Plex Mono'", "monospace"],
      },
      backgroundImage: {
        grain: "radial-gradient(circle at 1px 1px, rgba(255,255,255,0.035) 1px, transparent 0)",
      },
      keyframes: {
        stampIn: {
          "0%": { transform: "scale(2.2) rotate(-8deg)", opacity: "0" },
          "60%": { transform: "scale(0.95) rotate(-8deg)", opacity: "1" },
          "100%": { transform: "scale(1) rotate(-8deg)", opacity: "1" },
        },
        riseIn: {
          "0%": { transform: "translateY(10px)", opacity: "0" },
          "100%": { transform: "translateY(0)", opacity: "1" },
        },
        pulseDot: {
          "0%, 100%": { opacity: "0.4" },
          "50%": { opacity: "1" },
        },
      },
      animation: {
        stampIn: "stampIn 0.45s cubic-bezier(.2,.8,.3,1.2) forwards",
        riseIn: "riseIn 0.5s ease forwards",
        pulseDot: "pulseDot 1.6s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
