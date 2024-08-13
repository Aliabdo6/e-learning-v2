"use client";

import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";

interface SearchResult {
  title: string;
  category: string;
  course: string;
  lesson: string;
  url: string;
  snippet: string;
}

export default function SearchPage() {
  const searchParams = useSearchParams();
  const query = searchParams.get("q");
  const [results, setResults] = useState<
    SearchResult[]
  >([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchResults = async () => {
      setLoading(true);
      try {
        const response = await fetch(
          `/api/search?q=${encodeURIComponent(
            query || ""
          )}`
        );
        if (!response.ok) {
          throw new Error("Search failed");
        }
        const data = await response.json();
        setResults(data);
      } catch (error) {
        console.error(
          "Error fetching search results:",
          error
        );
      } finally {
        setLoading(false);
      }
    };

    if (query) {
      fetchResults();
    }
  }, [query]);

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6">
        Search Results for "{query}"
      </h1>
      {loading ? (
        <p>Loading...</p>
      ) : results.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {results.map((result, index) => (
            <Link href={result.url} key={index}>
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
                <h2 className="text-xl font-semibold mb-2">
                  {result.title}
                </h2>
                <p className="text-sm text-gray-600 dark:text-gray-300 mb-2">
                  {result.category} /{" "}
                  {result.course}
                </p>
                <p className="text-gray-700 dark:text-gray-200">
                  {result.snippet
                    .split(
                      new RegExp(
                        `(${query})`,
                        "gi"
                      )
                    )
                    .map((part, i) =>
                      part.toLowerCase() ===
                      query?.toLowerCase() ? (
                        <mark
                          key={i}
                          className="bg-yellow-200 dark:bg-yellow-800"
                        >
                          {part}
                        </mark>
                      ) : (
                        part
                      )
                    )}
                </p>
              </div>
            </Link>

            // <Link href={result.url} key={index}>
            //   <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
            //     <h2 className="text-xl font-semibold mb-2">
            //       {result.title}
            //     </h2>
            //     <p className="text-sm text-gray-600 dark:text-gray-300 mb-2">
            //       {result.category} /{" "}
            //       {result.course}
            //     </p>
            //     <p className="text-gray-700 dark:text-gray-200">
            //       {result.snippet}
            //     </p>
            //   </div>
            // </Link>
          ))}
        </div>
      ) : (
        <p>No results found.</p>
      )}
    </div>
  );
}
