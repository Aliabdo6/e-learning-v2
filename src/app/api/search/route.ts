import {
  getCourseCategories,
  getCourses,
  getLessons,
  getLessonContent,
} from "@/lib/api";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const query = searchParams.get("q");

  if (!query) {
    return new Response(
      JSON.stringify({
        error: "Query parameter is required",
      }),
      {
        status: 400,
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
  }

  const results = [];
  const categories = getCourseCategories();

  for (const category of categories) {
    const courses = getCourses(category);
    for (const course of courses) {
      const lessons = getLessons(
        category,
        course
      );
      for (const lesson of lessons) {
        const { data, content } =
          getLessonContent(
            category,
            course,
            lesson
          );
        if (
          data.title
            .toLowerCase()
            .includes(query.toLowerCase()) ||
          content
            .toLowerCase()
            .includes(query.toLowerCase())
        ) {
          results.push({
            title: data.title,
            category,
            course,
            lesson,
            url: `/courses/${category}/${course}/${lesson}`,
          });
        }
      }
    }
  }

  return new Response(JSON.stringify(results), {
    headers: {
      "Content-Type": "application/json",
    },
  });
}
