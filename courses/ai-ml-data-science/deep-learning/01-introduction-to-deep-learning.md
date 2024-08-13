### Building a Weather App with React and OpenWeatherMap API

Creating a weather app is a popular project that helps developers practice working with APIs and handling asynchronous data in React. In this tutorial, we’ll build a simple weather app that fetches data from the OpenWeatherMap API and displays the current weather for a user-specified location.

#### **What We’ll Build**

Our weather app will have the following features:

- A search bar to enter a city name.
- Display of the current weather conditions, including temperature, humidity, and a weather description.
- Error handling for invalid city names.

#### **Setting Up the Project**

Start by setting up a new React project using Create React App.

```bash
npx create-react-app weather-app
cd weather-app
npm start
```

This command sets up a new React project and starts the development server, which you can view in your browser at `http://localhost:3000`.

#### **Getting an API Key**

To fetch weather data, you’ll need to sign up for a free API key from [OpenWeatherMap](https://openweathermap.org/api). Once you have your API key, keep it handy, as we’ll use it in the code.

#### **Creating the Weather Component**

Let’s create a new component called `Weather.js` that will handle the main functionality of our app.

```javascript
import React, { useState } from "react";

function Weather() {
  const [city, setCity] = useState("");
  const [weather, setWeather] = useState(null);
  const [error, setError] = useState("");

  const API_KEY = "YOUR_API_KEY_HERE";

  const fetchWeather = async () => {
    if (city.trim() === "") {
      setError("Please enter a city name.");
      setWeather(null);
      return;
    }

    setError("");
    setWeather(null);

    try {
      const response = await fetch(
        `https://api.openweathermap.org/data/2.5/weather?q=${city}&appid=${API_KEY}&units=metric`
      );
      if (!response.ok) {
        throw new Error("City not found");
      }
      const data = await response.json();
      setWeather(data);
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div>
      <h1>Weather App</h1>
      <input
        type="text"
        value={city}
        onChange={(e) => setCity(e.target.value)}
        placeholder="Enter city name"
      />
      <button onClick={fetchWeather}>
        Get Weather
      </button>

      {error && (
        <p style={{ color: "red" }}>{error}</p>
      )}

      {weather && (
        <div>
          <h2>{weather.name}</h2>
          <p>
            Temperature: {weather.main.temp}°C
          </p>
          <p>
            Humidity: {weather.main.humidity}%
          </p>
          <p>{weather.weather[0].description}</p>
        </div>
      )}
    </div>
  );
}

export default Weather;
```

#### **Breaking Down the Code**

1. **State Management**:

   - We use `useState` to manage the city name entered by the user, the weather data fetched from the API, and any error messages.

2. **API Key**:

   - Replace `'YOUR_API_KEY_HERE'` with your actual OpenWeatherMap API key.

3. **Fetching Weather Data**:

   - The `fetchWeather` function is an asynchronous function that sends a request to the OpenWeatherMap API using the city name provided by the user.
   - If the city name is valid, it updates the `weather` state with the fetched data. If not, it sets an error message.

4. **Rendering**:
   - We render an input field and a button for the user to enter a city name and fetch the weather.
   - If there’s an error, we display it. If weather data is available, we display the temperature, humidity, and weather description.

#### **Integrating the Component**

Now that we’ve created our `Weather` component, let’s integrate it into our main application. Replace the contents of `App.js` with the following:

```javascript
import React from "react";
import Weather from "./Weather";
import "./App.css";

function App() {
  return (
    <div className="App">
      <Weather />
    </div>
  );
}

export default App;
```

Add some basic CSS to `App.css` for better styling:

```css
.App {
  text-align: center;
  margin-top: 50px;
}

input {
  padding: 10px;
  width: 200px;
  margin-right: 10px;
}

button {
  padding: 10px 15px;
}

div {
  margin-top: 20px;
}

h2 {
  margin-bottom: 10px;
}
```

#### **Running the App**

To run the app, use the command:

```bash
npm start
```

You can now search for the weather in different cities by entering the city name and clicking "Get Weather". The app will display the current weather information, and if the city is not found, an error message will appear.

#### **Conclusion**

In this tutorial, we built a simple weather app using React and the OpenWeatherMap API. This project is a great way to practice making API calls, managing state, and handling user input and errors in a React application. You can extend this app by adding features like displaying a 5-day weather forecast, incorporating geolocation to automatically show the weather for the user’s location, or improving the UI with more detailed weather data.
