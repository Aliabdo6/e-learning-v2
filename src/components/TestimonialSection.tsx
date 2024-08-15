export const TestimonialSection = () => {
  const testimonials = [
    {
      name: "John Doe",
      role: "Web Developer",
      quote:
        "This platform has been instrumental in advancing my career. The resources are top-notch!",
    },
    {
      name: "Jane Smith",
      role: "Data Scientist",
      quote:
        "I've learned more here in a few months than I did in years of traditional education.",
    },
    {
      name: "Alex Johnson",
      role: "Mobile App Developer",
      quote:
        "The community support and expert-led courses have been invaluable to my learning journey.",
    },
  ];

  return (
    <section className="container mx-auto px-4 py-6 ">
      <h2 className="text-3xl font-bold text-center mb-12 dark:text-white">
        What Our Learners Say
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        {testimonials.map(
          (testimonial, index) => (
            <div
              key={index}
              className="bg-white dark:bg-gray-700 rounded-lg p-6 shadow-md"
            >
              <p className="text-gray-600 dark:text-gray-300 mb-4 italic">
                "{testimonial.quote}"
              </p>
              <div className="flex items-center">
                <div className="w-12 h-12 bg-gray-300 rounded-full mr-4"></div>
                <div>
                  <h4 className="font-semibold dark:text-white">
                    {testimonial.name}
                  </h4>
                  <p className="text-gray-500 dark:text-gray-400">
                    {testimonial.role}
                  </p>
                </div>
              </div>
            </div>
          )
        )}
      </div>
    </section>
  );
};
