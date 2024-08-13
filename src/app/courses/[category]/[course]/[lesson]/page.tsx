import React from "react";
import Link from "next/link";
import {
  getLessonContent,
  getLessons,
  getCourseInfo,
} from "@/lib/api";
import Sidebar from "@/components/Sidebar";
import { remark } from "remark";
import html from "remark-html";
import { highlightCode } from "@/lib/prism";
import CopyButton from "@/components/CopyButton";

interface CodeBlockProps {
  node: any;
  value: string;
  lang: string;
}

const CodeBlock: React.FC<CodeBlockProps> = ({
  node,
  value,
  lang,
}) => {
  const highlightedCode = highlightCode(
    value || "",
    lang || "text"
  );
  return (
    <div className="relative">
      <pre className={`language-${lang}`}>
        <code
          dangerouslySetInnerHTML={{
            __html: highlightedCode,
          }}
        />
      </pre>
      <CopyButton code={value || ""} />
    </div>
  );
};
export default async function LessonPage({
  params,
}: {
  params: {
    category: string;
    course: string;
    lesson: string;
  };
}) {
  const { category, course, lesson } = params;
  const { data, content } = getLessonContent(
    category,
    course,
    lesson
  );
  const lessons = getLessons(category, course);
  const courseInfo = getCourseInfo(
    category,
    course
  );

  const processedContent = await remark()
    .use(html, {
      sanitize: false,
      createElement: React.createElement,
      components: {
        code: CodeBlock,
      },
    })
    .process(content);

  const contentHtml = processedContent.toString();

  const currentLessonIndex =
    lessons.indexOf(lesson);
  const prevLesson =
    currentLessonIndex > 0
      ? lessons[currentLessonIndex - 1]
      : null;
  const nextLesson =
    currentLessonIndex < lessons.length - 1
      ? lessons[currentLessonIndex + 1]
      : null;

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="bg-gray-100 dark:bg-gray-800 p-4 mb-8 rounded-lg">
        <h2 className="text-xl font-semibold">
          {courseInfo.title}
        </h2>
        <p className="text-gray-600 dark:text-gray-300">
          {data.title}
        </p>
      </div>
      <div className="flex flex-col md:flex-row">
        <Sidebar
          lessons={lessons}
          category={category}
          course={course}
          currentLesson={lesson}
        />
        <div className="flex-grow md:ml-8 mt-8 md:mt-0">
          <h1 className="text-3xl font-bold mb-4">
            {data.title}
          </h1>
          <div
            className="prose dark:prose-invert max-w-none"
            dangerouslySetInnerHTML={{
              __html: contentHtml,
            }}
          />
          <div className="mt-8 flex justify-between">
            {prevLesson && (
              <Link
                href={`/courses/${category}/${course}/${prevLesson}`}
                className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
              >
                Previous Lesson
              </Link>
            )}
            {nextLesson && (
              <Link
                href={`/courses/${category}/${course}/${nextLesson}`}
                className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
              >
                Next Lesson
              </Link>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// import React from "react";
// import {
//   getLessonContent,
//   getLessons,
// } from "@/lib/api";
// import Sidebar from "@/components/Sidebar";
// import { remark } from "remark";
// import html from "remark-html";
// import { highlightCode } from "@/lib/prism";
// import CopyButton from "@/components/CopyButton";

// const CodeBlock = ({ node, value, lang }) => {
//   const highlightedCode = highlightCode(
//     value || "",
//     lang || "text"
//   );
//   return (
//     <pre className={`language-${lang} relative`}>
//       <code
//         dangerouslySetInnerHTML={{
//           __html: highlightedCode,
//         }}
//       />
//       <CopyButton code={value || ""} />
//     </pre>
//   );
// };

// export default async function LessonPage({
//   params,
// }: {
//   params: {
//     category: string;
//     course: string;
//     lesson: string;
//   };
// }) {
//   const { category, course, lesson } = params;
//   const { data, content } = getLessonContent(
//     category,
//     course,
//     lesson
//   );
//   const lessons = getLessons(category, course);

//   const processedContent = await remark()
//     .use(html, {
//       sanitize: false,
//       createElement: React.createElement,
//       components: {
//         code: CodeBlock,
//       },
//     })
//     .process(content);

//   const contentHtml = processedContent.toString();

//   return (
//     <div className="container mx-auto px-4 py-8 flex">
//       <Sidebar
//         lessons={lessons}
//         category={category}
//         course={course}
//         currentLesson={lesson}
//       />
//       <div className="flex-grow ml-8">
//         <h1 className="text-3xl font-bold mb-4">
//           {data.title}
//         </h1>
//         <div
//           className="prose dark:prose-invert max-w-none"
//           dangerouslySetInnerHTML={{
//             __html: contentHtml,
//           }}
//         />
//       </div>
//     </div>
//   );
// }
