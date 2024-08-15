"use client";

import React, {
  useState,
  useEffect,
} from "react";
import Image from "next/image";

interface Book {
  id: number;
  title: string;
  author: string;
  description: string;
  image: string;
  pdfUrl: string;
}

export default function BooksPage() {
  const [books, setBooks] = useState<Book[]>([]);
  const [selectedBook, setSelectedBook] =
    useState<Book | null>(null);

  useEffect(() => {
    fetch("/books.json")
      .then((response) => response.json())
      .then((data) => setBooks(data));
  }, []);

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6">
        Available Books
      </h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {books.map((book) => (
          <BookCard
            key={book.id}
            book={book}
            onDetails={() =>
              setSelectedBook(book)
            }
          />
        ))}
      </div>
      {selectedBook && (
        <BookDetailsModal
          book={selectedBook}
          onClose={() => setSelectedBook(null)}
        />
      )}
    </div>
  );
}

function BookCard({
  book,
  onDetails,
}: {
  book: Book;
  onDetails: () => void;
}) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden">
      <Image
        src={book.image}
        alt={book.title}
        width={300}
        height={400}
        className="w-full h-48 object-cover"
      />
      <div className="p-4">
        <h2 className="text-xl font-semibold mb-2">
          {book.title}
        </h2>
        <p className="text-gray-600 dark:text-gray-300 mb-4">
          {book.author}
        </p>
        <div className="flex justify-between">
          <button
            onClick={onDetails}
            className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded"
            aria-label="View Details"
            title="View Details"
          >
            Details
          </button>
          <a
            href={book.pdfUrl}
            download
            className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded"
          >
            Download
          </a>
        </div>
      </div>
    </div>
  );
}

function BookDetailsModal({
  book,
  onClose,
}: {
  book: Book;
  onClose: () => void;
}) {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg max-w-2xl w-full">
        <div className="flex justify-between items-start p-4 border-b">
          <h2 className="text-2xl font-semibold">
            {book.title}
          </h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700"
            title="View Details"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>
        <div className="p-4 flex">
          <Image
            src={book.image}
            alt={book.title}
            width={200}
            height={300}
            className="object-cover mr-4"
          />
          <div>
            <p className="text-gray-600 dark:text-gray-300 mb-2">
              {book.author}
            </p>
            <p className="text-gray-800 dark:text-gray-200">
              {book.description}
            </p>
          </div>
        </div>
        <div className="p-4 border-t flex justify-end">
          <a
            href={book.pdfUrl}
            download
            className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded"
          >
            Download
          </a>
        </div>
      </div>
    </div>
  );
}
