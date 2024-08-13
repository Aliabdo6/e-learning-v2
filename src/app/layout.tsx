import type { Metadata } from "next";
import { ThemeProvider } from "next-themes";
import { Inter } from "next/font/google";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import "./globals.css";
import ScrollToTopButton from "@/components/ScrollToTopButton";
// import "../prism-theme.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title:
    "CodeSphere - Expand Your Coding Knowledge",
  description:
    "Learn programming with our comprehensive online courses",
  keywords:
    "coding, programming, online courses, education, web development",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
        >
          <div className="flex flex-col min-h-screen">
            <Header />
            <main className="flex-grow">
              {children}
            </main>
            <Footer />
            <ScrollToTopButton />
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
