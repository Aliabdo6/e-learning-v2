import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const category = searchParams.get("category");
  const quizName = searchParams.get("quiz");

  const quizzesDir = path.join(
    process.cwd(),
    "src",
    "quizzes"
  );

  if (!category && !quizName) {
    // Return all categories
    const categories = fs.readdirSync(quizzesDir);
    return NextResponse.json({ categories });
  }

  if (category && !quizName) {
    // Return all quizzes in a category
    const categoryDir = path.join(
      quizzesDir,
      category
    );
    const quizzes = fs
      .readdirSync(categoryDir)
      .map((file) => file.replace(".json", ""));
    return NextResponse.json({ quizzes });
  }

  if (category && quizName) {
    // Return specific quiz data
    const quizPath = path.join(
      quizzesDir,
      category,
      `${quizName}.json`
    );
    try {
      const quizData = JSON.parse(
        fs.readFileSync(quizPath, "utf8")
      );
      return NextResponse.json(quizData);
    } catch (error) {
      return NextResponse.json(
        { error: "Quiz not found" },
        { status: 404 }
      );
    }
  }

  return NextResponse.json(
    { error: "Invalid request" },
    { status: 400 }
  );
}
