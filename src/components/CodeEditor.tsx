import React from "react";

interface CodeEditorProps {
  value: string;
  onChange: (value: string) => void;
  language: string;
}

const CodeEditor: React.FC<CodeEditorProps> = ({
  value,
  onChange,
  language,
}) => {
  return (
    <textarea
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full h-64 p-2 font-mono text-sm border rounded"
      aria-label={`Code editor for ${language}`}
    />
  );
};

export default CodeEditor;
