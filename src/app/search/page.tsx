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
}

export default function SearchResults() {
  const searchParams = useSearchParams();
  const query = searchParams.get("q");
  const [results, setResults] = useState<
    SearchResult[]
  >([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<
    string | null
  >(null);

  useEffect(() => {
    const fetchResults = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(
          `/api/search?q=${encodeURIComponent(
            query || ""
          )}`
        );
        if (!response.ok)
          throw new Error(
            "Failed to fetch search results"
          );
        const data = await response.json();
        setResults(data);
      } catch (err) {
        setError(
          "An error occurred while fetching search results"
        );
      } finally {
        setLoading(false);
      }
    };

    if (query) {
      fetchResults();
    }
  }, [query]);

  if (loading)
    return (
      <div className="text-center py-8">
        Loading...
      </div>
    );
  if (error)
    return (
      <div className="text-center py-8 text-red-500">
        {error}
      </div>
    );

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6">
        Search Results for "{query}"
      </h1>
      {results.length === 0 ? (
        <p>No results found.</p>
      ) : (
        <ul className="space-y-4">
          {results.map((result, index) => (
            <li
              key={index}
              className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4"
            >
              <Link
                href={result.url}
                className="text-xl font-semibold hover:text-blue-500"
              >
                {result.title}
              </Link>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                {result.category} &gt;{" "}
                {result.course} &gt;{" "}
                {result.lesson}
              </p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
