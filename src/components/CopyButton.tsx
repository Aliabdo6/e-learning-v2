"use client";

import { useState } from "react";

export default function CopyButton({
  code,
}: {
  code: string;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button
      onClick={handleCopy}
      className="absolute top-2 right-2 bg-gray-200 dark:bg-gray-700 rounded px-2 py-1 text-sm"
    >
      {copied ? "Copied!" : "Copy"}
    </button>
  );
}
