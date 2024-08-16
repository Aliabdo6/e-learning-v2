### **Lesson 5: Introduction to Machine Learning**

#### **Objectives:**

1. Understand the basics of machine learning and its types.
2. Learn about common algorithms and their applications.
3. Implement a simple machine learning model.

#### **1. What is Machine Learning?**

Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve over time without being explicitly programmed. It involves creating algorithms that can recognize patterns and make decisions based on data.

**Key Concepts:**

- **Supervised Learning:** Algorithms learn from labeled data and make predictions based on input features (e.g., classification, regression).
- **Unsupervised Learning:** Algorithms find patterns and relationships in unlabeled data (e.g., clustering, dimensionality reduction).
- **Reinforcement Learning:** Algorithms learn by interacting with an environment and receiving feedback based on their actions.

#### **2. Common Machine Learning Algorithms**

- **Linear Regression:** Models the relationship between a dependent variable and one or more independent variables.
- **Logistic Regression:** Used for binary classification problems.
- **Decision Trees:** A tree-like model of decisions and their possible consequences.
- **k-Nearest Neighbors (k-NN):** Classifies data based on the closest training examples in the feature space.
- **Support Vector Machines (SVM):** Finds the optimal hyperplane that separates different classes.

**Code Example: Linear Regression**

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4], [5]])  # Independent variable
y = np.array([2, 3, 5, 7, 11])  # Dependent variable

# Creating and fitting the model
model = LinearRegression()
model.fit(X, y)

# Making predictions
predictions = model.predict(X)

print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_}")
print(f"Predictions: {predictions}")
```

#### **3. Implementing a Machine Learning Model**

- **Dataset Preparation:** Split data into training and testing sets.
- **Model Training:** Train the model using the training data.
- **Evaluation:** Assess model performance using metrics like accuracy, precision, recall, and F1-score.

**Code Example: Classification with k-Nearest Neighbors (k-NN)**

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
```

#### **4. Key Metrics for Model Evaluation**

- **Accuracy:** The proportion of correct predictions to total predictions.
- **Precision:** The proportion of true positive predictions to the total positive predictions.
- **Recall:** The proportion of true positive predictions to the total actual positives.
- **F1-Score:** The harmonic mean of precision and recall.

**Code Example: Evaluating a Classification Model**

```python
from sklearn.metrics import classification_report

# Print detailed classification report
print(classification_report(y_test, y_pred))
```

#### **5. Key Takeaways**

- Machine learning enables systems to learn from data and make predictions or decisions.
- Understanding different types of algorithms and their applications is crucial.
- Implementing and evaluating machine learning models involves dataset preparation, training, and performance assessment.

#### **6. Homework/Practice:**

- Implement and evaluate different machine learning algorithms on a dataset of your choice.
- Experiment with parameter tuning and model optimization techniques.
- Compare the performance of various models using appropriate metrics.
