import {
  getCourseCategories,
  getCourses,
  getLessons,
  getCourseInfo,
} from "@/lib/api";
import Link from "next/link";

export async function generateStaticParams() {
  const categories = getCourseCategories();
  const paths = [];

  for (const category of categories) {
    const courses = getCourses(category);
    for (const course of courses) {
      paths.push({ category, course });
    }
  }

  return paths;
}

export default function CoursePage({
  params,
}: {
  params: { category: string; course: string };
}) {
  const { category, course } = params;
  const lessons = getLessons(category, course);
  const { title, description } = getCourseInfo(
    category,
    course
  );

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-4">
        {title}
      </h1>
      <p className="text-lg mb-6">
        {description}
      </p>
      <h2 className="text-2xl font-semibold mb-4">
        Lessons
      </h2>
      <ul className="space-y-2">
        {lessons.map((lesson) => (
          <li key={lesson}>
            <Link
              href={`/courses/${category}/${course}/${lesson}`}
              className="text-blue-500 hover:underline"
            >
              {lesson}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
