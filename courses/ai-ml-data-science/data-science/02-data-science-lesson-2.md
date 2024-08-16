### **Lesson 2: Data Collection and Cleaning**

#### **Objectives:**

1. Understand different methods of data collection.
2. Learn how to clean and preprocess data for analysis.
3. Work with real-world datasets using Python.

#### **1. Methods of Data Collection**

Data can be collected from various sources, including:

- **Web Scraping:** Extracting data from websites.
- **APIs:** Using Application Programming Interfaces to retrieve data from online services.
- **Databases:** Querying data from SQL or NoSQL databases.
- **CSV/Excel Files:** Importing data from file formats commonly used for data storage.

**Code Example: Web Scraping with BeautifulSoup**

```python
import requests
from bs4 import BeautifulSoup

# Fetching a web page
response = requests.get('https://example.com')
soup = BeautifulSoup(response.content, 'html.parser')

# Extracting titles from the page
titles = [title.text for title in soup.find_all('h1')]
print(titles)
```

#### **2. Data Cleaning**

Data cleaning is crucial for ensuring that your data is accurate, consistent, and usable. Common tasks include:

- **Handling Missing Values:** Filling in, interpolating, or removing missing data.
- **Removing Duplicates:** Identifying and removing duplicate entries.
- **Data Transformation:** Converting data types and normalizing values.

**Code Example: Data Cleaning with Pandas**

```python
import pandas as pd

# Sample DataFrame with missing values and duplicates
data = {'Name': ['Alice', 'Bob', 'Charlie', 'Alice'],
        'Age': [24, 27, None, 24]}
df = pd.DataFrame(data)

# Handling missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Removing duplicates
df.drop_duplicates(inplace=True)

print(df)
```

#### **3. Data Preprocessing Techniques**

- **Normalization:** Scaling data to a range (e.g., 0 to 1).
- **Encoding Categorical Variables:** Converting categorical data into numerical format using techniques like one-hot encoding.
- **Feature Engineering:** Creating new features or modifying existing ones to improve model performance.

**Code Example: Encoding Categorical Variables**

```python
from sklearn.preprocessing import OneHotEncoder

# Sample DataFrame with categorical data
data = {'Color': ['Red', 'Green', 'Blue']}
df = pd.DataFrame(data)

# One-hot encoding
encoder = OneHotEncoder(sparse=False)
encoded_data = encoder.fit_transform(df[['Color']])

encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Color']))
print(encoded_df)
```

#### **4. Working with Real-World Datasets**

Practice loading and exploring datasets from real-world sources:

- **Kaggle Datasets:** [Kaggle](https://www.kaggle.com/datasets) offers a variety of datasets for practice.
- **UCI Machine Learning Repository:** [UCI](https://archive.ics.uci.edu/ml/index.php) provides numerous datasets for machine learning research.

**Code Example: Loading a CSV File**

```python
# Loading a CSV file
df = pd.read_csv('data.csv')

# Displaying the first few rows
print(df.head())

# Basic data exploration
print(df.info())
print(df.describe())
```

#### **5. Key Takeaways**

- Data collection methods vary based on the source and format of data.
- Effective data cleaning and preprocessing are critical steps before analysis or modeling.
- Familiarize yourself with tools and techniques to handle real-world data challenges.

#### **6. Homework/Practice:**

- Practice web scraping or use an API to collect a dataset.
- Clean and preprocess a dataset of your choice.
- Explore and visualize the dataset using Pandas and Matplotlib.
