import fs from "fs";
import path from "path";

const quizzesDir = path.join(
  process.cwd(),
  "src",
  "quizzes"
);

export async function getQuizCategories() {
  return fs.readdirSync(quizzesDir);
}

export async function getQuizzesByCategory(
  category: string
) {
  const categoryDir = path.join(
    quizzesDir,
    category
  );
  return fs
    .readdirSync(categoryDir)
    .map((file) => file.replace(".json", ""));
}

export async function getQuizData(
  category: string,
  quizName: string
) {
  const quizPath = path.join(
    quizzesDir,
    category,
    `${quizName}.json`
  );
  const quizData = JSON.parse(
    fs.readFileSync(quizPath, "utf8")
  );
  return quizData;
}
