import { CallToAction } from "@/components/CallToAction";
import CourseList from "@/components/CourseList";
import FeaturedSection from "@/components/FeaturedSection";
import { LearningPathways } from "@/components/LearningPathways";
import { TestimonialSection } from "@/components/TestimonialSection";
import {
  getCourseCategories,
  getCourses,
} from "@/lib/api";

export default function Home() {
  const categories = getCourseCategories();

  return (
    <>
      <div className="container mx-auto px-4 py-8">
        <div className=" text-center ">
          <h1 className="text-4xl font-bold mb-8">
            Welcome to our{" "}
            <span className="text-blue-400">
              Code
            </span>
            <span className="text-purple-400">
              Sphere
            </span>{" "}
            Platform
          </h1>
        </div>
        {categories.map((category) => (
          <section
            key={category}
            className="mb-12"
          >
            <h2 className="text-2xl font-semibold mb-4 capitalize">
              {category}
            </h2>
            <CourseList
              category={category}
              courses={getCourses(category)}
            />
          </section>
        ))}
      </div>
      <LearningPathways />
      <FeaturedSection />
      <TestimonialSection />
      <CallToAction />
    </>
  );
}
