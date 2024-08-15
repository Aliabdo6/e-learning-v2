import React from "react";

const UpcomingFeaturesSection: React.FC = () => {
  const features = [
    {
      title: "Certification System",
      description:
        "Earn official certificates upon completing courses to showcase your skills and knowledge.",
    },
    {
      title: "Badges System",
      description:
        "Collect badges for achieving milestones and demonstrating proficiency in various topics.",
    },
  ];

  return (
    <section className="py-20 bg-gray-100 dark:bg-gray-900">
      <div className="container mx-auto px-4">
        <h2 className="text-4xl font-bold text-center mb-12 text-text-light dark:text-text-dark transform transition duration-500 hover:scale-105">
          Coming Soon
        </h2>
        <div className="grid md:grid-cols-2 gap-10">
          {features.map((feature, index) => (
            <div
              key={index}
              className="bg-white dark:bg-gray-800 p-8 rounded-xl shadow-lg hover:shadow-2xl transition duration-300 transform hover:scale-105"
            >
              <h3 className="text-2xl font-semibold mb-4 text-primary-light dark:text-primary-dark">
                {feature.title}
              </h3>
              <p className="text-gray-600 dark:text-gray-300">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default UpcomingFeaturesSection;
