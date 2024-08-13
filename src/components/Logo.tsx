import Link from "next/link";
import React from "react";

const Logo = () => {
  return (
    <Link href="/">
      <div className="flex items-center space-x-2 cursor-pointer group">
        <div className="relative w-10 h-10">
          <div className="absolute inset-0 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg transform rotate-45 group-hover:rotate-180 transition-transform duration-500 ease-in-out"></div>
          <div className="absolute inset-0 bg-white bg-opacity-30 rounded-lg transform rotate-45 scale-75 group-hover:scale-110 transition-transform duration-500 ease-in-out"></div>
          <span className="absolute inset-0 flex items-center justify-center text-white font-bold text-xl">
            🌐
          </span>
        </div>
        <div className="text-white font-bold text-xl tracking-wider">
          <span className="text-blue-400">
            Code
          </span>
          <span className="text-purple-400">
            Sphere
          </span>
        </div>
      </div>
    </Link>
  );
};

export default Logo;
