// src/app/quizzes/[category]/page.tsx

import React from "react";
import Link from "next/link";
import { getQuizzesByCategory } from "../../../lib/quizzes";

export default async function QuizCategoryPage({
  params,
}: {
  params: { category: string };
}) {
  const quizzes = await getQuizzesByCategory(
    params.category
  );

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6 capitalize">
        {params.category} Quizzes
      </h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {quizzes.map((quiz) => (
          <Link
            key={quiz}
            href={`/quizzes/${params.category}/${quiz}`}
          >
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
              <h3 className="text-xl font-semibold mb-2">
                {quiz}
              </h3>
              <p className="text-gray-600 dark:text-gray-300">
                Take this quiz to test your{" "}
                {params.category} knowledge
              </p>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
