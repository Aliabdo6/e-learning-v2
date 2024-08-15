export const LearningPathways = () => {
  const pathways = [
    {
      title: "Web Development",
      icon: "🌐",
      description:
        "Master modern web technologies and frameworks",
    },
    {
      title: "Data Science",
      icon: "📊",
      description:
        "Dive into data analysis, machine learning, and AI",
    },
    {
      title: "Mobile App Development",
      icon: "📱",
      description:
        "Create cutting-edge mobile applications",
    },
    {
      title: "Cloud Computing",
      icon: "☁️",
      description:
        "Explore cloud platforms and serverless architecture",
    },
  ];

  return (
    <section className="bg-gray-100 dark:bg-gray-800 py-16">
      <div className="container mx-auto px-4">
        <h2 className="text-3xl font-bold text-center mb-12 dark:text-white">
          Learning Pathways
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {pathways.map((pathway, index) => (
            <div
              key={index}
              className="bg-white dark:bg-gray-700 rounded-lg p-6 shadow-md transition-all duration-300 hover:shadow-xl"
            >
              <div className="text-4xl mb-4">
                {pathway.icon}
              </div>
              <h3 className="text-xl font-semibold mb-2 dark:text-white">
                {pathway.title}
              </h3>
              <p className="text-gray-600 dark:text-gray-300">
                {pathway.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};
