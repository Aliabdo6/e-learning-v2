import Logo from "./Logo";

export default function Footer() {
  return (
    <footer className="bg-gray-100 dark:bg-gray-900">
      <div className="container mx-auto px-4 py-6">
        <div className="flex flex-col md:flex-row items-center justify-between">
          <Logo />
          <p className="text-gray-600 dark:text-gray-400 mt-4 md:mt-0">
            © {new Date().getFullYear()}{" "}
            CodeSphere. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
}
