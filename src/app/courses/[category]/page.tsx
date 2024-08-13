import {
  getCourseCategories,
  getCourses,
} from "@/lib/api";
import CourseList from "@/components/CourseList";

export async function generateStaticParams() {
  const categories = getCourseCategories();
  return categories.map((category) => ({
    category,
  }));
}

export default function CategoryPage({
  params,
}: {
  params: { category: string };
}) {
  const { category } = params;
  const courses = getCourses(category);

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6 capitalize">
        {category} Courses
      </h1>
      <CourseList
        category={category}
        courses={courses}
      />
    </div>
  );
}
