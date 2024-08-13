import { NextResponse } from "next/server";
import { auth } from "@clerk/nextjs/server";
import dbConnect from "@/lib/mongodb";
import Post from "@/models/Post";

export async function GET() {
  await dbConnect();
  const posts = await Post.find().sort({
    createdAt: -1,
  });
  return NextResponse.json(posts);
}

export async function POST(request: Request) {
  const { userId } = auth();
  if (!userId) {
    return NextResponse.json(
      { error: "Unauthorized" },
      { status: 401 }
    );
  }

  await dbConnect();
  const { title, content } = await request.json();
  const post = await Post.create({
    title,
    content,
    author: userId,
  });
  return NextResponse.json(post);
}
