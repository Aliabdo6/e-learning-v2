// src/app/quizzes/page.tsx

import React from "react";
import Link from "next/link";
import { getQuizCategories } from "../../lib/quizzes";
import QuizCategoryCard from "@/components/QuizCategoryCard";

export default async function QuizzesPage() {
  const categories = await getQuizCategories();

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-8">
        Quiz Categories
      </h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {categories.map((category) => (
          <QuizCategoryCard
            key={category}
            category={category}
          />
        ))}
      </div>
      <section className="mt-12">
        <h2 className="text-2xl font-semibold mb-4">
          About Our Quiz System
        </h2>
        <p className="mb-4">
          Our quiz system is designed to help you
          test and improve your knowledge across
          various categories. With multiple
          question types including
          multiple-choice, coding challenges, and
          open-ended questions, you'll find a
          diverse range of engaging content to
          enhance your learning experience.
        </p>
        <Link
          href="/quizzes/about"
          className="text-blue-500 hover:underline"
        >
          Learn more about our quiz system
        </Link>
      </section>
      <section className="mt-12">
        <h2 className="text-2xl font-semibold mb-4">
          Discussion
        </h2>
        <p className="mb-4">
          Join the conversation about our quizzes!
          Share your thoughts, ask questions, and
          connect with other learners.
        </p>
        {/* <Link
          href="/quizzes/discussion"
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        >
          Go to Discussion Forum
        </Link> */}
      </section>
    </div>
  );
}
