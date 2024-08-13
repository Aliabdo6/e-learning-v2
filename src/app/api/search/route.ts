import { NextResponse } from "next/server";
import { buildSearchIndex } from "@/lib/api";

const searchIndex = buildSearchIndex();

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const query = searchParams.get("q");

  if (!query) {
    return NextResponse.json(
      { error: "Query parameter is required" },
      { status: 400 }
    );
  }

  const queryWords = query
    .toLowerCase()
    .split(/\W+/);
  const results = searchIndex
    .map((item) => {
      const matchCount = queryWords.reduce(
        (count, word) => {
          return (
            count +
            (item.words.includes(word) ? 1 : 0)
          );
        },
        0
      );

      return {
        ...item,
        matchCount,
        relevance: matchCount / queryWords.length,
      };
    })
    .filter((item) => item.matchCount > 0)
    .sort((a, b) => b.relevance - a.relevance)
    .map(
      ({
        title,
        category,
        course,
        lesson,
        url,
        content,
      }) => ({
        title,
        category,
        course,
        lesson,
        url,
        snippet: getSnippet(
          content,
          queryWords[0]
        ),
      })
    );

  return NextResponse.json(results.slice(0, 20)); // Limit to top 20 results
}

function getSnippet(
  content: string,
  keyword: string
) {
  const index = content
    .toLowerCase()
    .indexOf(keyword.toLowerCase());
  if (index === -1)
    return content.substring(0, 150) + "...";

  const start = Math.max(0, index - 75);
  const end = Math.min(
    content.length,
    index + 75
  );
  return (
    (start > 0 ? "..." : "") +
    content.substring(start, end) +
    (end < content.length ? "..." : "")
  );
}
