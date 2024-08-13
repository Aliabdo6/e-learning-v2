"use client";

import { useState, useEffect } from "react";
import { useUser } from "@clerk/nextjs";

interface Post {
  _id: string;
  title: string;
  content: string;
  author: string;
  createdAt: string;
}

interface Comment {
  _id: string;
  content: string;
  author: string;
  createdAt: string;
}

export default function PostPage({
  params,
}: {
  params: { postId: string };
}) {
  const { user, isLoaded } = useUser();
  const [post, setPost] = useState<Post | null>(
    null
  );
  const [comments, setComments] = useState<
    Comment[]
  >([]);
  const [newComment, setNewComment] =
    useState("");

  useEffect(() => {
    fetchPost();
    fetchComments();
  }, []);

  const fetchPost = async () => {
    const response = await fetch(
      `/api/posts/${params.postId}`
    );
    const data = await response.json();
    setPost(data);
  };

  const fetchComments = async () => {
    const response = await fetch(
      `/api/posts/${params.postId}/comments`
    );
    const data = await response.json();
    setComments(data);
  };

  const handleSubmit = async (
    e: React.FormEvent
  ) => {
    e.preventDefault();
    if (!user) return;

    const response = await fetch(
      `/api/posts/${params.postId}/comments`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          content: newComment,
        }),
      }
    );

    if (response.ok) {
      setNewComment("");
      fetchComments();
    }
  };

  if (!isLoaded || !post)
    return <div>Loading...</div>;

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6">
        {post.title}
      </h1>
      <p className="text-gray-600 dark:text-gray-300 mb-4">
        {post.content}
      </p>
      <p className="text-sm text-gray-500 mb-8">
        Posted by {post.author} on{" "}
        {new Date(
          post.createdAt
        ).toLocaleString()}
      </p>

      <h2 className="text-2xl font-semibold mb-4">
        Comments
      </h2>

      {user && (
        <form
          onSubmit={handleSubmit}
          className="mb-8"
        >
          <textarea
            placeholder="Add a comment"
            value={newComment}
            onChange={(e) =>
              setNewComment(e.target.value)
            }
            className="w-full p-2 mb-4 border rounded"
            required
          />
          <button
            type="submit"
            className="bg-blue-500 text-white px-4 py-2 rounded"
          >
            Post Comment
          </button>
        </form>
      )}

      <div className="space-y-4">
        {comments.map((comment) => (
          <div
            key={comment._id}
            className="bg-gray-100 dark:bg-gray-700 p-4 rounded"
          >
            <p className="mb-2">
              {comment.content}
            </p>
            <p className="text-sm text-gray-500">
              Commented by {comment.author} on{" "}
              {new Date(
                comment.createdAt
              ).toLocaleString()}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
