import { NextResponse } from "next/server";
import { auth } from "@clerk/nextjs/server";
import dbConnect from "@/lib/mongodb";
import Comment from "@/models/Comment";

export async function GET(
  request: Request,
  { params }: { params: { postId: string } }
) {
  await dbConnect();
  const comments = await Comment.find({
    postId: params.postId,
  }).sort({ createdAt: -1 });
  return NextResponse.json(comments);
}

export async function POST(
  request: Request,
  { params }: { params: { postId: string } }
) {
  const { userId } = auth();
  if (!userId) {
    return NextResponse.json(
      { error: "Unauthorized" },
      { status: 401 }
    );
  }

  await dbConnect();
  const { content } = await request.json();
  const comment = await Comment.create({
    content,
    author: userId,
    postId: params.postId,
  });
  return NextResponse.json(comment);
}
