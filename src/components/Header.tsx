"use client";

import Link from "next/link";
import { useTheme } from "next-themes";
import SearchBar from "./SearchBar";
import { useEffect, useState } from "react";

export default function Header() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  // After mounting, we have access to the theme
  useEffect(() => setMounted(true), []);

  return (
    <>
      <header className="bg-white dark:bg-gray-800 shadow-md">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <Link
            href="/"
            className="text-2xl font-bold text-blue-600 dark:text-blue-400"
          >
            eLearning Platform
          </Link>
          <div className="flex items-center space-x-4">
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
        </div>
      </header>
    </>
  );
}

// "use client";

// import Link from "next/link";
// import { useTheme } from "next-themes";
// import SearchBar from "./SearchBar";

// export default function Header() {
//   const { theme, setTheme } = useTheme();

//   return (
//     <>
//       <header className="bg-white dark:bg-gray-800 shadow-md">
//         <div className="container mx-auto px-4 py-4 flex items-center justify-between">
//           <Link
//             href="/"
//             className="text-2xl font-bold text-blue-600 dark:text-blue-400"
//           >
//             eLearning Platform
//           </Link>
//           <div className="flex items-center space-x-4">
//             <SearchBar />
//             <button
//               onClick={() =>
//                 setTheme(
//                   theme === "dark"
//                     ? "light"
//                     : "dark"
//                 )
//               }
//               className="p-2 rounded-full bg-gray-200 dark:bg-gray-700"
//             >
//               {theme === "dark" ? "🌞" : "🌙"}
//             </button>
//           </div>
//         </div>
//       </header>
//     </>
//   );
// }
