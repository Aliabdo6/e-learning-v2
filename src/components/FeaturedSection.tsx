import React from "react";
import Link from "next/link";
import Image from "next/image";

const featuredItems = [
  {
    title: "Interactive Quizzes",
    description:
      "Test your knowledge with our engaging quizzes across various topics.",
    image: "/images/quezes.jpg",
    link: "/quizzes",
  },
  {
    title: "E-Book Library",
    description:
      "Access our extensive collection of e-books to enhance your learning.",
    image: "/images/ebook.jpg",
    link: "/books",
  },
  {
    title: "Learning Community",
    description:
      "Join our vibrant community on Discord, Telegram, and WhatsApp.",
    image: "/images/community.jpg",
    link: "/community",
  },
];

const FeaturedSection = () => {
  return (
    <section className="py-16 bg-gray-100 dark:bg-gray-900">
      <div className="container mx-auto px-4">
        <h2 className="text-3xl font-bold text-center mb-12 dark:text-white">
          Featured Resources
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {featuredItems.map((item, index) => (
            <div
              key={index}
              className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden transform transition duration-300 hover:scale-105"
            >
              <Image
                src={item.image}
                alt={item.title}
                width={400}
                height={250}
                className="w-full h-48 object-cover"
              />
              <div className="p-6">
                <h3 className="text-xl font-semibold mb-2 dark:text-white">
                  {item.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-300 mb-4">
                  {item.description}
                </p>
                <Link
                  href={item.link}
                  className="inline-block bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition duration-300"
                >
                  Explore
                </Link>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FeaturedSection;
