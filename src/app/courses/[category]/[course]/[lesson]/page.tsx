import React from "react";
import {
  getLessonContent,
  getLessons,
} from "@/lib/api";
import Sidebar from "@/components/Sidebar";
import { remark } from "remark";
import html from "remark-html";
import { highlightCode } from "@/lib/prism";
import CopyButton from "@/components/CopyButton";

const CodeBlock = ({ node, value, lang }) => {
  const highlightedCode = highlightCode(
    value || "",
    lang || "text"
  );
  return (
    <pre className={`language-${lang} relative`}>
      <code
        dangerouslySetInnerHTML={{
          __html: highlightedCode,
        }}
      />
      <CopyButton code={value || ""} />
    </pre>
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

  return (
    <div className="container mx-auto px-4 py-8 flex">
      <Sidebar
        lessons={lessons}
        category={category}
        course={course}
        currentLesson={lesson}
      />
      <div className="flex-grow ml-8">
        <h1 className="text-3xl font-bold mb-4">
          {data.title}
        </h1>
        <div
          className="prose dark:prose-invert max-w-none"
          dangerouslySetInnerHTML={{
            __html: contentHtml,
          }}
        />
      </div>
    </div>
  );
}

// import {
//   getLessonContent,
//   getLessons,
// } from "@/lib/api";
// import Sidebar from "@/components/Sidebar";
// import { remark } from "remark";
// import html from "remark-html";
// import { highlightCode } from "@/lib/prism";
// // import CopyButton from "@/components/CopyButton";

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
//       handlers: {
//         code(h, node) {
//           const value = node.value
//             ? node.value.toString()
//             : "";
//           const lang = node.lang || "text";
//           const highlightedCode = highlightCode(
//             value,
//             lang
//           );
//           const props = {
//             className: `language-${lang}`,
//           };
//           return h(node, "pre", props, [
//             h(
//               node,
//               "code",
//               props,
//               highlightedCode
//             ),
//           ]);
//         },
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

// import {
//   getLessonContent,
//   getLessons,
// } from "@/lib/api";
// import Sidebar from "@/components/Sidebar";
// import { remark } from "remark";
// import html from "remark-html";

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
//     .use(html)
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
