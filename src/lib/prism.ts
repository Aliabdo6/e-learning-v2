import Prism from "prismjs";
import "prismjs/components/prism-javascript";
import "prismjs/components/prism-typescript";
import "prismjs/components/prism-python";
import "prismjs/components/prism-java";
import "prismjs/components/prism-css";
import "prismjs/components/prism-jsx";

export function highlightCode(
  code: string,
  language: string
) {
  return Prism.highlight(
    code,
    Prism.languages[language],
    language
  );
}
