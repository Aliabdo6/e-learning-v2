import React from "react";
import Link from "next/link";

export default function AboutQuizSystem() {
  return (
    <div className="container mx-auto px-4 py-8 dark:bg-gray-800 dark:text-white">
      <h1 className="text-3xl font-bold mb-6">
        About Our Quiz System
      </h1>
      <div className="prose max-w-none dark:prose-invert">
        <p>
          Our quiz system is a comprehensive
          platform designed to enhance your
          learning experience through interactive
          and engaging quizzes across various
          categories.
        </p>
        <h2>Key Features</h2>
        <ul>
          <li>
            Multiple question types:
            multiple-choice, coding challenges,
            and open-ended questions
          </li>
          <li>
            Dynamic quiz generation based on JSON
            files
          </li>
          <li>
            Category-based organization for easy
            navigation
          </li>
          <li>
            Immediate feedback on quiz performance
          </li>
          <li>Adaptive difficulty levels</li>
        </ul>
        <h2>How It Works</h2>
        <p>
          Our system dynamically generates quizzes
          based on JSON files stored in our
          database. This approach allows us to
          easily add new quizzes and categories,
          ensuring that our content stays fresh
          and up-to-date.
        </p>
        <h2>Benefits</h2>
        <ul>
          <li>
            Reinforces learning through active
            recall
          </li>
          <li>
            Provides a fun and interactive way to
            test your knowledge
          </li>
          <li>
            Helps identify areas for improvement
          </li>
          <li>Tracks progress over time</li>
        </ul>
        <p>
          Start exploring our quizzes today and
          take your learning to the next level!
        </p>
        <Link
          href="/quizzes"
          className="text-blue-500 hover:underline"
        >
          Back to Quiz Categories
        </Link>
      </div>
    </div>
  );
}

// import React from "react";
// import Link from "next/link";

// export default function AboutQuizSystem() {
//   return (
//     <div className="container mx-auto px-4 py-8">
//       <h1 className="text-3xl font-bold mb-6">
//         About Our Quiz System
//       </h1>
//       <div className="prose max-w-none">
//         <p>
//           Our quiz system is a comprehensive
//           platform designed to enhance your
//           learning experience through interactive
//           and engaging quizzes across various
//           categories.
//         </p>
//         <h2>Key Features</h2>
//         <ul>
//           <li>
//             Multiple question types:
//             multiple-choice, coding challenges,
//             and open-ended questions
//           </li>
//           <li>
//             Dynamic quiz generation based on JSON
//             files
//           </li>
//           <li>
//             Category-based organization for easy
//             navigation
//           </li>
//           <li>
//             Immediate feedback on quiz performance
//           </li>
//           <li>
//             Discussion forum for community
//             engagement
//           </li>
//         </ul>
//         <h2>How It Works</h2>
//         <p>
//           Our system dynamically generates quizzes
//           based on JSON files stored in our
//           database. This approach allows us to
//           easily add new quizzes and categories,
//           ensuring that our content stays fresh
//           and up-to-date.
//         </p>
//         <h2>Benefits</h2>
//         <ul>
//           <li>
//             Reinforces learning through active
//             recall
//           </li>
//           <li>
//             Provides a fun and interactive way to
//             test your knowledge
//           </li>
//           <li>
//             Helps identify areas for improvement
//           </li>
//           <li>
//             Encourages community engagement and
//             discussion
//           </li>
//         </ul>
//         <p>
//           Start exploring our quizzes today and
//           take your learning to the next level!
//         </p>
//         <Link
//           href="/quizzes"
//           className="text-blue-500 hover:underline"
//         >
//           Back to Quiz Categories
//         </Link>
//       </div>
//     </div>
//   );
// }
