"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { useTheme } from "next-themes";
import SearchBar from "./SearchBar";
import Logo from "./Logo";

export default function Header() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [prevScrollPos, setPrevScrollPos] =
    useState(0);
  const [visible, setVisible] = useState(true);

  useEffect(() => setMounted(true), []);

  useEffect(() => {
    const handleScroll = () => {
      const currentScrollPos = window.pageYOffset;
      setVisible(
        prevScrollPos > currentScrollPos ||
          currentScrollPos < 10
      );
      setPrevScrollPos(currentScrollPos);
    };

    window.addEventListener(
      "scroll",
      handleScroll
    );
    return () =>
      window.removeEventListener(
        "scroll",
        handleScroll
      );
  }, [prevScrollPos]);

  return (
    <header
      className={`sticky top-0 z-50 bg-white dark:bg-gray-800 shadow-md transition-transform duration-300 ${
        visible
          ? "translate-y-0"
          : "-translate-y-full"
      }`}
    >
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <Logo />
          <div className="hidden md:flex items-center space-x-4">
            <SearchBar />
            {mounted && (
              <button
                onClick={() =>
                  setTheme(
                    theme === "dark"
                      ? "light"
                      : "dark"
                  )
                }
                className="p-2 rounded-full bg-gray-200 dark:bg-gray-700"
              >
                {theme === "dark" ? "🌞" : "🌙"}
              </button>
            )}
          </div>
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="md:hidden p-2"
          >
            {isOpen ? "✕" : "☰"}
          </button>
        </div>
        {isOpen && (
          <div className="mt-4 md:hidden">
            <SearchBar />
            {mounted && (
              <button
                onClick={() =>
                  setTheme(
                    theme === "dark"
                      ? "light"
                      : "dark"
                  )
                }
                className="mt-4 p-2 rounded-full bg-gray-200 dark:bg-gray-700 w-full"
              >
                {theme === "dark"
                  ? "Switch to Light Mode"
                  : "Switch to Dark Mode"}
              </button>
            )}
          </div>
        )}
      </div>
    </header>
  );
}

// "use client";

// import Link from "next/link";
// import { useTheme } from "next-themes";
// import SearchBar from "./SearchBar";
// import { useEffect, useState } from "react";
// import Logo from "./Logo";

// export default function Header() {
//   const { theme, setTheme } = useTheme();
//   const [mounted, setMounted] = useState(false);

//   // After mounting, we have access to the theme
//   useEffect(() => setMounted(true), []);

//   return (
//     <>
//       <header className="bg-white dark:bg-gray-800 shadow-md">
//         <div className="container mx-auto px-4 py-4 flex items-center justify-between">
//           <Link
//             href="/"
//             className="text-2xl font-bold text-blue-600 dark:text-blue-400"
//           >
//             <Logo />
//           </Link>
//           <div className="flex items-center space-x-4">
//             <SearchBar />
//             {mounted && (
//               <button
//                 onClick={() =>
//                   setTheme(
//                     theme === "dark"
//                       ? "light"
//                       : "dark"
//                   )
//                 }
//                 className="p-2 rounded-full bg-gray-200 dark:bg-gray-700"
//               >
//                 {theme === "dark" ? "🌞" : "🌙"}
//               </button>
//             )}
//           </div>
//         </div>
//       </header>
//     </>
//   );
// }
