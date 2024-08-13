import CourseCard from "./CourseCard";
import { getCourseInfo } from "@/lib/api";

interface CourseListProps {
  category: string;
  courses: string[];
}

export default function CourseList({
  category,
  courses,
}: CourseListProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {courses.map((course) => {
        const { title, description } =
          getCourseInfo(category, course);
        return (
          <CourseCard
            key={course}
            title={title}
            description={description}
            category={category}
            slug={course}
          />
        );
      })}
    </div>
  );
}
