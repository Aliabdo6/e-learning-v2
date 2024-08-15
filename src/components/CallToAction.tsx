import Link from "next/link";

export const CallToAction = () => {
  return (
    <section className="bg-blue-600 dark:bg-blue-800 py-16">
      <div className="container mx-auto px-4 text-center">
        <h2 className="text-3xl font-bold text-white mb-4">
          Ready to Start Your Learning Journey?
        </h2>
        <p className="text-xl text-blue-100 mb-8">
          Join thousands of learners and take the
          first step towards mastering new skills.
        </p>
        <Link
          href="/community"
          className="bg-white text-blue-600 px-8 py-3 rounded-full font-semibold text-lg hover:bg-blue-100 transition duration-300"
        >
          Join to our Community
        </Link>
      </div>
    </section>
  );
};
