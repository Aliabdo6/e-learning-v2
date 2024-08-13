"use client";

import { useState, useEffect } from "react";
import { useUser } from "@clerk/nextjs";
import Link from "next/link";

interface Post {
  _id: string;
  title: string;
  content: string;
  author: string;
  createdAt: string;
}

export default function ForumPage() {
  const { user, isLoaded } = useUser();
  const [posts, setPosts] = useState<Post[]>([]);
  const [newPost, setNewPost] = useState({
    title: "",
    content: "",
  });

  useEffect(() => {
    fetchPosts();
  }, []);

  const fetchPosts = async () => {
    const response = await fetch("/api/posts");
    const data = await response.json();
    setPosts(data);
  };

  const handleSubmit = async (
    e: React.FormEvent
  ) => {
    e.preventDefault();
    if (!user) return;

    const response = await fetch("/api/posts", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(newPost),
    });

    if (response.ok) {
      setNewPost({ title: "", content: "" });
      fetchPosts();
    }
  };

  if (!isLoaded) return <div>Loading...</div>;

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6">
        Forum
      </h1>

      {user && (
        <form
          onSubmit={handleSubmit}
          className="mb-8"
        >
          <input
            type="text"
            placeholder="Title"
            value={newPost.title}
            onChange={(e) =>
              setNewPost({
                ...newPost,
                title: e.target.value,
              })
            }
            className="w-full p-2 mb-4 border rounded"
            required
          />
          <textarea
            placeholder="Content"
            value={newPost.content}
            onChange={(e) =>
              setNewPost({
                ...newPost,
                content: e.target.value,
              })
            }
            className="w-full p-2 mb-4 border rounded"
            required
          />
          <button
            type="submit"
            className="bg-blue-500 text-white px-4 py-2 rounded"
          >
            Create Post
          </button>
        </form>
      )}

      <div className="space-y-6">
        {posts.map((post) => (
          <div
            key={post._id}
            className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow"
          >
            <Link href={`/forum/${post._id}`}>
              <h2 className="text-xl font-semibold mb-2">
                {post.title}
              </h2>
            </Link>
            <p className="text-gray-600 dark:text-gray-300 mb-2">
              {post.content}
            </p>
            <p className="text-sm text-gray-500">
              Posted by {post.author} on{" "}
              {new Date(
                post.createdAt
              ).toLocaleString()}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
