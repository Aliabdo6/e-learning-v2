# Codesphere One

Codesphere One is a modern and interactive e-learning platform designed to provide users with a seamless educational experience. The platform leverages the latest web technologies to deliver dynamic content and user-friendly interfaces.

## Live Demo

Check out the live version of the application here: [Codesphere One](https://codesphere-one.vercel.app/)

## Features

- **User Authentication**: Secure and easy user login with Clerk integration.
- **Responsive Design**: Built with Tailwind CSS for mobile-first responsiveness.
- **Dynamic Content**: Supports markdown-based content with syntax highlighting using PrismJS.
- **Dark Mode**: Theme switching enabled via `next-themes`.
- **Animation**: Smooth transitions and animations powered by Framer Motion.
- **Content Management**: Easily manage and parse markdown content with Gray Matter, Remark, and Rehype.

## Tech Stack

- **Next.js**: A powerful React framework for server-rendered applications.
- **React**: A JavaScript library for building user interfaces.
- **Tailwind CSS**: A utility-first CSS framework for rapid UI development.
- **MongoDB & Mongoose**: For handling data storage and schema-based data validation.
- **Clerk**: Authentication solution for Next.js.
- **PrismJS**: Syntax highlighting for code blocks in markdown.
- **Framer Motion**: Library for creating animations and transitions in React.
- **Remark & Rehype**: Tools for parsing and converting markdown content.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Aliabdo6/e-learning-v2.git
   cd e-learning-v2
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Environment Variables**: 
   Create a `.env.local` file in the root directory and add your MongoDB connection string, Clerk API keys, and other necessary environment variables.

4. **Run the development server**:
   ```bash
   npm run dev
   ```
   Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Deployment

The app is currently deployed on Vercel. You can deploy your own version by connecting the GitHub repository to your Vercel account and following their deployment instructions.

## Contributing

Feel free to open issues or submit pull requests to improve the app. Contributions are welcome!

## License

This project is licensed under the MIT License.

## Contact

For more information, you can contact [Ali Abdo](https://github.com/Aliabdo6).

