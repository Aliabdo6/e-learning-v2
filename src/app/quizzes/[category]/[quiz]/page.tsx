"use client";

import React, {
  useState,
  useEffect,
  useRef,
} from "react";
import { useParams } from "next/navigation";
import QuizQuestion from "@/components/QuizQuestion";

interface QuizData {
  title: string;
  questions: Question[];
}

interface Question {
  id: number;
  type: string;
  question: string;
  options?: string[];
  correctAnswer: any;
  initialCode?: string;
  solution?: string;
  testCases?: { input: any[]; expected: any }[];
  sampleAnswer?: string;
  feedback: string;
}

export default function QuizPage() {
  const params = useParams();
  const [quizData, setQuizData] =
    useState<QuizData | null>(null);
  const [
    currentQuestionIndex,
    setCurrentQuestionIndex,
  ] = useState(0);
  const [userAnswers, setUserAnswers] = useState<
    Record<number, any>
  >({});
  const [
    answeredQuestions,
    setAnsweredQuestions,
  ] = useState<Record<number, boolean>>({});
  const feedbackRef =
    useRef<HTMLDivElement>(null);

  useEffect(() => {
    const fetchQuizData = async () => {
      const response = await fetch(
        `/api/quizzes?category=${params.category}&quiz=${params.quiz}`
      );
      const data = await response.json();
      setQuizData(data);
    };
    fetchQuizData();
  }, [params]);

  const handleAnswer = (
    answer: any,
    isCorrect: boolean
  ) => {
    setUserAnswers({
      ...userAnswers,
      [currentQuestionIndex]: answer,
    });
    setAnsweredQuestions({
      ...answeredQuestions,
      [currentQuestionIndex]: isCorrect,
    });
    if (isCorrect) {
      setTimeout(() => {
        feedbackRef.current?.scrollIntoView({
          behavior: "smooth",
        });
      }, 100);
    }
  };

  const handleNext = () => {
    if (
      quizData &&
      currentQuestionIndex <
        quizData.questions.length - 1
    ) {
      setCurrentQuestionIndex(
        currentQuestionIndex + 1
      );
    }
  };

  const handlePrevious = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(
        currentQuestionIndex - 1
      );
    }
  };

  if (!quizData)
    return (
      <div className="dark:text-white">
        Loading...
      </div>
    );

  const currentQuestion =
    quizData.questions[currentQuestionIndex];
  const isAnswered =
    answeredQuestions[currentQuestionIndex] ||
    false;

  return (
    <div className="container mx-auto px-4 py-8 dark:bg-gray-800 dark:text-white">
      <h1 className="text-3xl font-bold mb-4">
        {quizData.title}
      </h1>
      <p className="mb-4">
        Question {currentQuestionIndex + 1} of{" "}
        {quizData.questions.length}
      </p>
      <QuizQuestion
        question={currentQuestion}
        onAnswer={handleAnswer}
        userAnswer={
          userAnswers[currentQuestionIndex]
        }
        isAnswered={isAnswered}
      />
      {isAnswered && (
        <div
          ref={feedbackRef}
          className="mt-4 p-4 bg-green-100 dark:bg-green-800 rounded"
        >
          <h3 className="font-bold">Feedback:</h3>
          <p>{currentQuestion.feedback}</p>
        </div>
      )}
      <div className="mt-4 flex justify-between">
        <button
          onClick={handlePrevious}
          disabled={currentQuestionIndex === 0}
          className="bg-primary-light dark:bg-primary-dark text-white px-4 py-2 rounded disabled:opacity-50"
        >
          Previous
        </button>
        <button
          onClick={handleNext}
          disabled={
            currentQuestionIndex ===
            quizData.questions.length - 1
          }
          className="bg-primary-light dark:bg-primary-dark text-white px-4 py-2 rounded disabled:opacity-50"
        >
          Next
        </button>
      </div>
    </div>
  );
}
