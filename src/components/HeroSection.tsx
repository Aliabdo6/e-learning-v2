import React from "react";

const HeroSection = () => {
  return (
    <section className="bg-background-light dark:bg-background-dark text-text-light dark:text-text-dark py-16 px-4 md:px-8">
      <div className="max-w-7xl mx-auto text-center">
        <h1 className="text-4xl font-bold mb-4">
          Welcome to our CodeSphere Platform
        </h1>
        <p className="text-lg mb-8">
          Explore courses in AI/ML, Data Science,
          Web Development, and more!
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div className="bg-primary-light dark:bg-primary-dark p-4 border border-primary-dark dark:border-primary-light rounded-lg shadow-md transform transition duration-300 hover:scale-105 hover:shadow-lg">
            <h2 className="text-xl font-semibold mb-2">
              Deep Learning
            </h2>
            <p>
              Explore neural networks and deep
              learning models.
            </p>
          </div>

          <div className="bg-primary-light dark:bg-primary-dark p-4 border border-primary-dark dark:border-primary-light rounded-lg shadow-md transform transition duration-300 hover:scale-105 hover:shadow-lg">
            <h2 className="text-xl font-semibold mb-2">
              Machine Learning
            </h2>
            <p>
              Introduction to machine learning
              concepts, algorithms, and tools.
            </p>
          </div>

          <div className="bg-primary-light dark:bg-primary-dark p-4 border border-primary-dark dark:border-primary-light rounded-lg shadow-md transform transition duration-300 hover:scale-105 hover:shadow-lg">
            <h2 className="text-xl font-semibold mb-2">
              Algorithms
            </h2>
            <p>
              Study fundamental algorithms used in
              computer science.
            </p>
          </div>

          <div className="bg-primary-light dark:bg-primary-dark p-4 border border-primary-dark dark:border-primary-light rounded-lg shadow-md transform transition duration-300 hover:scale-105 hover:shadow-lg">
            <h2 className="text-xl font-semibold mb-2">
              Data Structures
            </h2>
            <p>
              Learn about different data
              structures and their applications.
            </p>
          </div>

          <div className="bg-primary-light dark:bg-primary-dark p-4 border border-primary-dark dark:border-primary-light rounded-lg shadow-md transform transition duration-300 hover:scale-105 hover:shadow-lg">
            <h2 className="text-xl font-semibold mb-2">
              NoSQL Databases
            </h2>
            <p>
              Understand NoSQL databases like
              MongoDB and their use cases.
            </p>
          </div>

          <div className="bg-primary-light dark:bg-primary-dark p-4 border border-primary-dark dark:border-primary-light rounded-lg shadow-md transform transition duration-300 hover:scale-105 hover:shadow-lg">
            <h2 className="text-xl font-semibold mb-2">
              SQL Databases
            </h2>
            <p>
              Explore SQL and relational databases
              for data storage and management.
            </p>
          </div>

          <div className="bg-primary-light dark:bg-primary-dark p-4 border border-primary-dark dark:border-primary-light rounded-lg shadow-md transform transition duration-300 hover:scale-105 hover:shadow-lg">
            <h2 className="text-xl font-semibold mb-2">
              CI/CD
            </h2>
            <p>
              Implement Continuous Integration and
              Continuous Deployment pipelines.
            </p>
          </div>

          <div className="bg-primary-light dark:bg-primary-dark p-4 border border-primary-dark dark:border-primary-light rounded-lg shadow-md transform transition duration-300 hover:scale-105 hover:shadow-lg">
            <h2 className="text-xl font-semibold mb-2">
              Docker
            </h2>
            <p>
              Learn about containerization and how
              to use Docker for application
              deployment.
            </p>
          </div>

          <div className="bg-primary-light dark:bg-primary-dark p-4 border border-primary-dark dark:border-primary-light rounded-lg shadow-md transform transition duration-300 hover:scale-105 hover:shadow-lg">
            <h2 className="text-xl font-semibold mb-2">
              Java
            </h2>
            <p>
              Learn Java programming for software
              development and enterprise
              applications.
            </p>
          </div>

          <div className="bg-primary-light dark:bg-primary-dark p-4 border border-primary-dark dark:border-primary-light rounded-lg shadow-md transform transition duration-300 hover:scale-105 hover:shadow-lg">
            <h2 className="text-xl font-semibold mb-2">
              JavaScript
            </h2>
            <p>
              Understand JavaScript for building
              dynamic web applications.
            </p>
          </div>

          <div className="bg-primary-light dark:bg-primary-dark p-4 border border-primary-dark dark:border-primary-light rounded-lg shadow-md transform transition duration-300 hover:scale-105 hover:shadow-lg">
            <h2 className="text-xl font-semibold mb-2">
              Python
            </h2>
            <p>
              Master Python programming for
              various applications including data
              science and web development.
            </p>
          </div>

          <div className="bg-primary-light dark:bg-primary-dark p-4 border border-primary-dark dark:border-primary-light rounded-lg shadow-md transform transition duration-300 hover:scale-105 hover:shadow-lg">
            <h2 className="text-xl font-semibold mb-2">
              Git
            </h2>
            <p>
              Master Git version control for
              collaborative software development.
            </p>
          </div>

          <div className="bg-primary-light dark:bg-primary-dark p-4 border border-primary-dark dark:border-primary-light rounded-lg shadow-md transform transition duration-300 hover:scale-105 hover:shadow-lg">
            <h2 className="text-xl font-semibold mb-2">
              Linux
            </h2>
            <p>
              Learn Linux command line and shell
              scripting.
            </p>
          </div>

          <div className="bg-primary-light dark:bg-primary-dark p-4 border border-primary-dark dark:border-primary-light rounded-lg shadow-md transform transition duration-300 hover:scale-105 hover:shadow-lg">
            <h2 className="text-xl font-semibold mb-2">
              Backend Development
            </h2>
            <p>
              Understand server-side programming
              and database interactions.
            </p>
          </div>

          <div className="bg-primary-light dark:bg-primary-dark p-4 border border-primary-dark dark:border-primary-light rounded-lg shadow-md transform transition duration-300 hover:scale-105 hover:shadow-lg">
            <h2 className="text-xl font-semibold mb-2">
              Frontend Development
            </h2>
            <p>
              Learn to build responsive web
              interfaces using HTML, CSS, and
              JavaScript.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
