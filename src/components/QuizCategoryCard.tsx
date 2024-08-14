// src/components/QuizCategoryCard.tsx

import React from "react";
import Link from "next/link";

interface QuizCategoryCardProps {
  category: string;
}

const QuizCategoryCard: React.FC<
  QuizCategoryCardProps
> = ({ category }) => {
  return (
    <Link href={`/quizzes/${category}`}>
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
        <h3 className="text-xl font-semibold mb-2 capitalize">
          {category}
        </h3>
        <p className="text-gray-600 dark:text-gray-300">
          Test your knowledge in {category}
        </p>
      </div>
    </Link>
  );
};

export default QuizCategoryCard;
