"use client";

import React, { useState } from "react";

const FAQSection: React.FC = () => {
  const faqs = [
    {
      question:
        "What courses does CodeSphere offer?",
      answer:
        "CodeSphere offers courses in AI/ML, cloud computing, computer science, cybersecurity, databases, DevOps, game development, mobile development, programming languages, and web development.",
    },
    {
      question: "Is CodeSphere really free?",
      answer:
        "Yes, CodeSphere is completely free. You can access all courses, quizzes, and community features at no cost.",
    },
    {
      question: "How do I get started?",
      answer:
        "Simply sign up for an account on our platform, choose a course that interests you, and start learning at your own pace.",
    },
  ];

  const [activeIndex, setActiveIndex] = useState<
    number | null
  >(null);

  return (
    <section className="py-20 bg-background-light dark:bg-background-dark">
      <div className="container mx-auto px-4">
        <h2 className="text-4xl font-bold text-center mb-12 text-text-light dark:text-text-dark transform transition duration-500 hover:scale-105">
          Frequently Asked Questions
        </h2>
        <div className="max-w-3xl mx-auto">
          {faqs.map((faq, index) => (
            <div
              key={index}
              className="mb-6 transform transition duration-500 hover:translate-y-[-5px]"
            >
              <button
                onClick={() =>
                  setActiveIndex(
                    activeIndex === index
                      ? null
                      : index
                  )
                }
                className="flex justify-between items-center w-full text-left p-4 bg-white dark:bg-gray-800 rounded-lg shadow-md hover:shadow-lg transition duration-300"
              >
                <h3 className="text-xl font-semibold text-primary-light dark:text-primary-dark">
                  {faq.question}
                </h3>
                <svg
                  className={`w-6 h-6 transition-transform duration-300 ${
                    activeIndex === index
                      ? "transform rotate-180"
                      : ""
                  }`}
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
                    clipRule="evenodd"
                  />
                </svg>
              </button>
              <div
                className={`mt-2 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg transition-all duration-300 ease-in-out ${
                  activeIndex === index
                    ? "max-h-96 opacity-100"
                    : "max-h-0 opacity-0 overflow-hidden"
                }`}
              >
                <p className="text-gray-600 dark:text-gray-300">
                  {faq.answer}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FAQSection;
