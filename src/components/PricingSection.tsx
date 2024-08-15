import React from "react";

const PricingSection: React.FC = () => {
  const features = [
    "Access to all courses",
    "Interactive quizzes",
    "Community access",
  ];

  return (
    <section className="py-20 bg-gradient-to-b from-background-light to-gray-100 dark:from-background-dark dark:to-gray-900">
      <div className="container mx-auto px-4">
        <h2 className="text-4xl font-bold text-center mb-12 text-text-light dark:text-text-dark transform transition duration-500 hover:scale-105">
          Pricing
        </h2>
        <div className="max-w-md mx-auto bg-white dark:bg-gray-800 rounded-2xl shadow-2xl overflow-hidden transform hover:scale-105 transition duration-300">
          <div className="p-8">
            <h3 className="text-3xl font-semibold text-center text-primary-light dark:text-primary-dark mb-6">
              Free Forever
            </h3>
            <p className="text-center text-gray-600 dark:text-gray-300 mb-8">
              Enjoy all features of CodeSphere at
              no cost!
            </p>
            <ul className="mb-8 space-y-4">
              {features.map((feature, index) => (
                <li
                  key={index}
                  className="flex items-center transform transition duration-500 hover:translate-x-2"
                >
                  <svg
                    className="w-5 h-5 mr-3 text-green-500"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                      clipRule="evenodd"
                    />
                  </svg>
                  <span>{feature}</span>
                </li>
              ))}
            </ul>
            <button className="w-full bg-primary-light dark:bg-primary-dark text-white py-3 px-6 rounded-lg text-lg font-semibold hover:bg-blue-600 dark:hover:bg-blue-500 transition duration-300 transform hover:scale-105 active:scale-95">
              Get Started
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default PricingSection;
