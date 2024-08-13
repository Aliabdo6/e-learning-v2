import Link from "next/link";

interface SidebarProps {
  lessons: string[];
  category: string;
  course: string;
  currentLesson: string;
}

export default function Sidebar({
  lessons,
  category,
  course,
  currentLesson,
}: SidebarProps) {
  return (
    <aside className="w-64 bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
      <h2 className="text-xl font-semibold mb-4">
        Lessons
      </h2>
      <ul className="space-y-2">
        {lessons.map((lesson) => (
          <li key={lesson}>
            <Link
              href={`/courses/${category}/${course}/${lesson}`}
              className={`block p-2 rounded ${
                lesson === currentLesson
                  ? "bg-blue-500 text-white"
                  : "hover:bg-gray-200 dark:hover:bg-gray-700"
              }`}
            >
              {lesson}
            </Link>
          </li>
        ))}
      </ul>
    </aside>
  );
}
