### Machine Learning Course - Lesson 3: Supervised Learning Algorithms

#### Lesson Overview

In this lesson, we will explore various supervised learning algorithms, including both classification and regression methods. We will cover the theory behind these algorithms and provide code examples to illustrate their implementation and usage.

#### 1. Classification Algorithms

Classification algorithms are used when the output variable is a category (e.g., spam or not spam). Here, we’ll cover some popular classification algorithms: Logistic Regression, Decision Trees, and k-Nearest Neighbors (k-NN).

##### **1.1 Logistic Regression**

Logistic Regression is a simple algorithm used for binary classification problems. It models the probability of a binary outcome based on one or more input features.

**Example Code (Python):**

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Use only two classes for binary classification
X = X[y != 2]
y = y[y != 2]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

##### **1.2 Decision Trees**

Decision Trees are a versatile algorithm that can be used for both classification and regression. They work by splitting the data into subsets based on feature values to make decisions.

**Example Code (Python):**

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load and prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

##### **1.3 k-Nearest Neighbors (k-NN)**

k-NN is a non-parametric algorithm that classifies a data point based on the majority class among its k nearest neighbors.

**Example Code (Python):**

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load and prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

#### 2. Regression Algorithms

Regression algorithms are used when the output variable is continuous. We’ll cover Linear Regression, Decision Tree Regression, and Support Vector Regression.

##### **2.1 Linear Regression**

Linear Regression is a simple algorithm used for predicting a continuous target variable based on one or more input features.

**Example Code (Python):**

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Sample dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 2, 1.3, 3.75, 2.25])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))
```

##### **2.2 Decision Tree Regression**

Decision Tree Regression works similarly to classification but is used for predicting continuous values.

**Example Code (Python):**

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Train model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))
```

##### **2.3 Support Vector Regression (SVR)**

SVR is a regression algorithm that uses support vector machines to find a function that deviates from the actual observed values by a value less than a specified margin.

**Example Code (Python):**

```python
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Train model
model = SVR(kernel='linear')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))
```

#### 3. Summary

In this lesson, we covered several supervised learning algorithms for both classification and regression tasks. We provided code examples for Logistic Regression, Decision Trees, k-NN, Linear Regression, Decision Tree Regression, and SVR.

In the next lesson, we will explore unsupervised learning algorithms and techniques, including clustering and dimensionality reduction.
