import React from "react";

const TimelineSection: React.FC = () => {
  const stages = [
    {
      title: "Fundamentals",
      description:
        "Learn the basics of programming and computer science.",
    },
    {
      title: "Intermediate Concepts",
      description:
        "Dive deeper into specific technologies and frameworks.",
    },
    {
      title: "Advanced Topics",
      description:
        "Master complex algorithms and system design.",
    },
    {
      title: "Specialization",
      description:
        "Focus on your chosen field and build real-world projects.",
    },
  ];

  return (
    <section className="py-20 bg-gradient-to-b from-background-light to-gray-100 dark:from-background-dark dark:to-gray-900">
      <div className="container mx-auto px-4">
        <h2 className="text-4xl font-bold text-center mb-12 text-text-light dark:text-text-dark transform transition duration-500 hover:scale-105">
          Your Learning Journey
        </h2>
        <div className="relative">
          {stages.map((stage, index) => (
            <div
              key={index}
              className={`flex ${
                index % 2 === 0
                  ? "justify-start"
                  : "justify-end"
              } mb-8 transform transition duration-500 hover:translate-y-[-5px]`}
            >
              <div className="w-1/2 p-6 bg-white dark:bg-gray-800 rounded-xl shadow-lg">
                <h3 className="text-2xl font-semibold mb-2 text-primary-light dark:text-primary-dark">
                  {stage.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-300">
                  {stage.description}
                </p>
              </div>
            </div>
          ))}
          <div className="absolute top-0 bottom-0 left-1/2 w-1 bg-primary-light dark:bg-primary-dark"></div>
        </div>
      </div>
    </section>
  );
};

export default TimelineSection;
