## JavaScript Tutorial for Beginners

JavaScript is a versatile and powerful programming language that is essential for web development. This tutorial will cover the basics of JavaScript, including syntax, variables, functions, and basic DOM manipulation.

### Table of Contents

1. Introduction to JavaScript
2. Setting Up Your Environment
3. Basic Syntax
4. Variables
5. Data Types
6. Operators
7. Conditional Statements
8. Loops
9. Functions
10. Arrays
11. Objects
12. Basic DOM Manipulation

### 1. Introduction to JavaScript

JavaScript is a high-level, interpreted programming language that is widely used to make web pages interactive. It can update and change both HTML and CSS, and it can calculate, manipulate, and validate data.

### 2. Setting Up Your Environment

To write and run JavaScript code, you'll need a text editor and a web browser. Popular text editors include VS Code, Sublime Text, and Atom.

### 3. Basic Syntax

JavaScript code is written in plain text and can be embedded in HTML using the `<script>` tag.

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>JavaScript Basics</title>
  </head>
  <body>
    <h1>Hello, World!</h1>
    <script>
      console.log("Hello, World!");
    </script>
  </body>
</html>
```

### 4. Variables

Variables store data values and can be declared using `var`, `let`, or `const`.

```javascript
var name = "John"; // global or function-scoped
let age = 25; // block-scoped
const pi = 3.14; // block-scoped and cannot be reassigned
```

### 5. Data Types

JavaScript supports several data types:

- **Number**: `let number = 42;`
- **String**: `let text = 'Hello, World!';`
- **Boolean**: `let isTrue = true;`
- **Array**: `let fruits = ['Apple', 'Banana', 'Cherry'];`
- **Object**: `let person = { name: 'John', age: 25 };`
- **Null**: `let emptyValue = null;`
- **Undefined**: `let notAssigned;`

### 6. Operators

Operators are used to perform operations on variables and values.

```javascript
let sum = 10 + 5; // Addition
let difference = 10 - 5; // Subtraction
let product = 10 * 5; // Multiplication
let quotient = 10 / 5; // Division
let remainder = 10 % 3; // Modulus
```

### 7. Conditional Statements

Conditional statements are used to perform different actions based on different conditions.

```javascript
let age = 18;

if (age >= 18) {
  console.log("You are an adult.");
} else {
  console.log("You are a minor.");
}
```

### 8. Loops

Loops are used to execute a block of code repeatedly.

```javascript
// For loop
for (let i = 0; i < 5; i++) {
  console.log("Iteration " + i);
}

// While loop
let count = 0;
while (count < 5) {
  console.log("Count " + count);
  count++;
}
```

### 9. Functions

Functions are blocks of code designed to perform a particular task.

```javascript
function greet(name) {
  return "Hello, " + name;
}

console.log(greet("John"));
```

### 10. Arrays

Arrays are used to store multiple values in a single variable.

```javascript
let colors = ["Red", "Green", "Blue"];

console.log(colors[0]); // Output: Red
colors.push("Yellow"); // Add an element
console.log(colors.length); // Output: 4
```

### 11. Objects

Objects are collections of key-value pairs.

```javascript
let car = {
  make: "Toyota",
  model: "Corolla",
  year: 2020,
};

console.log(car.make); // Output: Toyota
car.color = "Red"; // Add a new property
```

### 12. Basic DOM Manipulation

JavaScript can be used to manipulate the DOM (Document Object Model) to change the content of web pages.

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>DOM Manipulation</title>
  </head>
  <body>
    <h1 id="title">Original Title</h1>
    <button onclick="changeTitle()">
      Change Title
    </button>

    <script>
      function changeTitle() {
        document.getElementById(
          "title"
        ).textContent = "New Title";
      }
    </script>
  </body>
</html>
```

### Conclusion

This tutorial covers the basics of JavaScript. As you continue learning, you'll discover more advanced topics like asynchronous programming, ES6 features, and JavaScript frameworks and libraries like React.js, Angular, and Vue.js. Happy coding!
