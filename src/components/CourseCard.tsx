import Link from "next/link";

interface CourseCardProps {
  title: string;
  description: string;
  category: string;
  slug: string;
}

export default function CourseCard({
  title,
  description,
  category,
  slug,
}: CourseCardProps) {
  return (
    <Link href={`/courses/${category}/${slug}`}>
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
        <h3 className="text-xl font-semibold mb-2">
          {title}
        </h3>
        <p className="text-gray-600 dark:text-gray-300">
          {description}
        </p>
      </div>
    </Link>
  );
}
