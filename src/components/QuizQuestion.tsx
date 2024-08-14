"use client";

import React, {
  useState,
  useEffect,
} from "react";
import CodeEditor from "./CodeEditor";

interface QuizQuestionProps {
  question: {
    type: string;
    question: string;
    options?: string[];
    initialCode?: string;
    correctAnswer: any;
    feedback: string;
  };
  onAnswer: (
    answer: any,
    isCorrect: boolean
  ) => void;
  userAnswer: any;
  isAnswered: boolean;
}

const QuizQuestion: React.FC<
  QuizQuestionProps
> = ({
  question,
  onAnswer,
  userAnswer,
  isAnswered,
}) => {
  const [answer, setAnswer] = useState<any>(
    question.type === "code"
      ? question.initialCode || ""
      : ""
  );
  const [feedback, setFeedback] = useState<
    string | null
  >(null);

  useEffect(() => {
    if (isAnswered) {
      setFeedback(question.feedback);
    }
  }, [isAnswered, question.feedback]);

  const handleSubmit = () => {
    let isCorrect = false;
    switch (question.type) {
      case "multiple-choice":
        isCorrect =
          answer === question.correctAnswer;
        break;
      case "code":
        // Here you would typically run the code through a test suite
        // For this example, we'll just check if it's exactly the same as the correct answer
        isCorrect =
          answer === question.correctAnswer;
        break;
      case "text":
        // For text answers, you might want to do a case-insensitive comparison
        isCorrect =
          answer.toLowerCase().trim() ===
          question.correctAnswer
            .toLowerCase()
            .trim();
        break;
    }

    onAnswer(answer, isCorrect);
  };

  const renderQuestion = () => {
    switch (question.type) {
      case "multiple-choice":
        return (
          <ul>
            {question.options?.map(
              (option, index) => (
                <li key={index} className="mb-2">
                  <button
                    onClick={() =>
                      setAnswer(index)
                    }
                    disabled={isAnswered}
                    className={`w-full text-left p-2 rounded ${
                      answer === index
                        ? "bg-primary-light dark:bg-primary-dark text-white"
                        : "bg-gray-200 dark:bg-gray-700"
                    } ${
                      isAnswered
                        ? "opacity-50 cursor-not-allowed"
                        : ""
                    }`}
                  >
                    {option}
                  </button>
                </li>
              )
            )}
          </ul>
        );
      case "code":
        return (
          <CodeEditor
            value={answer}
            onChange={setAnswer}
            language="javascript"
          />
        );
      case "text":
        return (
          <textarea
            value={answer}
            onChange={(e) =>
              setAnswer(e.target.value)
            }
            className="w-full h-32 p-2 border rounded dark:bg-gray-700 dark:text-white"
            placeholder="Type your answer here..."
            readOnly={isAnswered}
          />
        );
      default:
        return null;
    }
  };

  return (
    <div className="mb-8">
      <h2 className="text-xl font-semibold mb-4 dark:text-white">
        {question.question}
      </h2>
      {renderQuestion()}
      <button
        onClick={handleSubmit}
        disabled={isAnswered}
        className={`mt-4 bg-primary-light dark:bg-primary-dark text-white px-4 py-2 rounded hover:opacity-90 ${
          isAnswered
            ? "opacity-50 cursor-not-allowed"
            : ""
        }`}
      >
        Submit Answer
      </button>
      {feedback && (
        <p
          className={`mt-2 ${
            isAnswered
              ? "text-green-500"
              : "text-red-500"
          }`}
        >
          {feedback}
        </p>
      )}
    </div>
  );
};

export default QuizQuestion;
