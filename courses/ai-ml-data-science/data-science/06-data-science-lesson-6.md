### **Lesson 6: Supervised Learning – Regression and Classification**

#### **Objectives:**

1. Understand and differentiate between regression and classification tasks.
2. Learn about key algorithms for regression and classification.
3. Implement and evaluate regression and classification models.

---

#### **1. Supervised Learning Overview**

Supervised learning involves training a model on a labeled dataset, where the outcome is known. The goal is to make predictions or classifications based on input data.

**Types:**

- **Regression:** Predicts continuous values (e.g., predicting house prices).
- **Classification:** Predicts discrete labels or categories (e.g., email spam detection).

---

#### **2. Regression Analysis**

Regression models predict a continuous output based on input features. Common regression algorithms include:

**a. Linear Regression**
Linear regression models the relationship between the dependent variable and one or more independent variables using a linear equation.

**Code Example: Simple Linear Regression**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample Data
X = np.array([[1], [2], [3], [4], [5]])  # Independent variable
y = np.array([2, 3, 5, 7, 11])  # Dependent variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R^2 Score: {r2_score(y_test, y_pred)}")

# Plotting
plt.scatter(X_test, y_test, color='black', label='Actual data')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()
```

**b. Polynomial Regression**
Polynomial regression extends linear regression by fitting a polynomial curve to the data.

**Code Example: Polynomial Regression**

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Transform features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Create and train model
model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R^2 Score: {r2_score(y_test, y_pred)}")

# Plotting
plt.scatter(X, y, color='black', label='Actual data')
X_fit = np.linspace(min(X), max(X), 100).reshape(-1, 1)
plt.plot(X_fit, model.predict(poly.transform(X_fit)), color='blue', linewidth=3, label='Fitted curve')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.legend()
plt.show()
```

---

#### **3. Classification Analysis**

Classification models predict categorical labels. Common classification algorithms include:

**a. Logistic Regression**
Logistic regression is used for binary classification problems, predicting probabilities that can be mapped to classes.

**Code Example: Logistic Regression**

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
```

**b. k-Nearest Neighbors (k-NN)**
k-NN classifies data based on the majority label among its k nearest neighbors.

**Code Example: k-NN Classification**

```python
from sklearn.neighbors import KNeighborsClassifier

# Create and train model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
```

---

#### **4. Model Evaluation Metrics**

- **For Regression:**

  - **Mean Squared Error (MSE):** Measures average squared error between predicted and actual values.
  - **R-squared (R²):** Indicates the proportion of variance explained by the model.

- **For Classification:**
  - **Accuracy:** The proportion of correctly classified instances.
  - **Confusion Matrix:** A matrix showing true positives, true negatives, false positives, and false negatives.
  - **Precision, Recall, F1-Score:** Metrics to evaluate classification performance.

**Code Example: Model Evaluation Metrics**

```python
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# For regression
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R^2 Score: {r2_score(y_test, y_pred)}")

# For classification
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
```

---

#### **5. Key Takeaways**

- **Regression** models predict continuous values and are evaluated using metrics like MSE and R².
- **Classification** models predict categorical outcomes and are evaluated using accuracy, confusion matrix, and classification report.
- Implementing and evaluating models requires understanding both the algorithms and appropriate metrics.

#### **6. Homework/Practice:**

- Choose a dataset and implement both regression and classification models.
- Evaluate and compare the performance of different algorithms using appropriate metrics.
- Experiment with hyperparameter tuning and model optimization techniques to improve performance.
