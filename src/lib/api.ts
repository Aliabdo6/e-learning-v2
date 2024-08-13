import fs from "fs";
import path from "path";
import matter from "gray-matter";

const coursesDirectory = path.join(
  process.cwd(),
  "courses"
);

export function getCourseCategories() {
  return fs.readdirSync(coursesDirectory);
}

export function getCourses(category: string) {
  const categoryPath = path.join(
    coursesDirectory,
    category
  );
  return fs.readdirSync(categoryPath);
}

export function getLessons(
  category: string,
  course: string
) {
  const coursePath = path.join(
    coursesDirectory,
    category,
    course
  );
  return fs
    .readdirSync(coursePath)
    .filter((file) => file.endsWith(".md"))
    .map((file) => file.replace(/\.md$/, ""));
}

export function getLessonContent(
  category: string,
  course: string,
  lesson: string
) {
  const filePath = path.join(
    coursesDirectory,
    category,
    course,
    `${lesson}.md`
  );
  const fileContents = fs.readFileSync(
    filePath,
    "utf8"
  );
  const { data, content } = matter(fileContents);
  return { data, content };
}

export function getCourseInfo(
  category: string,
  course: string
) {
  const coursePath = path.join(
    coursesDirectory,
    category,
    course,
    "info.json"
  );
  const fileContents = fs.readFileSync(
    coursePath,
    "utf8"
  );
  return JSON.parse(fileContents);
}

export function buildSearchIndex() {
  const index = [];
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
        const words = content
          .toLowerCase()
          .split(/\W+/);
        index.push({
          title: data.title,
          category,
          course,
          lesson,
          url: `/courses/${category}/${course}/${lesson}`,
          content: content,
          words: words,
        });
      }
    }
  }

  return index;
}
