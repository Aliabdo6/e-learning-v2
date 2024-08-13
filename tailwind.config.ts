import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          light: "#3B82F6", // blue-500
          dark: "#60A5FA", // blue-400
        },
        background: {
          light: "#FFFFFF",
          dark: "#1F2937", // gray-800
        },
        text: {
          light: "#1F2937", // gray-800
          dark: "#F3F4F6", // gray-100
        },
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
  darkMode: "class",
};

export default config;
