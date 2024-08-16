### Machine Learning Course - Lesson 5: Model Evaluation and Selection

#### Lesson Overview

In this lesson, we will focus on evaluating and selecting machine learning models. Understanding how to assess model performance is crucial for building effective and reliable models. We will cover evaluation metrics for classification and regression tasks, cross-validation techniques, and hyperparameter tuning.

#### 1. Evaluation Metrics

##### **1.1 Classification Metrics**

For classification tasks, the following metrics are commonly used to evaluate model performance:

- **Accuracy**: The proportion of correctly classified instances out of the total instances.
- **Precision**: The proportion of true positive instances among all instances classified as positive.
- **Recall (Sensitivity)**: The proportion of true positive instances among all actual positive instances.
- **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two.

**Example Code (Python):**

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Sample predictions
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)
```

##### **1.2 Regression Metrics**

For regression tasks, the following metrics are used to evaluate model performance:

- **Mean Absolute Error (MAE)**: The average of the absolute errors between predicted and actual values.
- **Mean Squared Error (MSE)**: The average of the squared errors between predicted and actual values.
- **Root Mean Squared Error (RMSE)**: The square root of the mean squared error, providing error magnitude in the original units.
- **R-squared (R²)**: The proportion of variance in the dependent variable that is predictable from the independent variables.

**Example Code (Python):**

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Sample predictions
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

# Calculate metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)
```

#### 2. Cross-Validation

Cross-validation is a technique for assessing the performance of a model by splitting the data into multiple folds. It helps in evaluating how well the model generalizes to unseen data.

##### **2.1 K-Fold Cross-Validation**

In K-Fold Cross-Validation, the dataset is divided into \( K \) subsets (folds). The model is trained on \( K-1 \) folds and tested on the remaining fold. This process is repeated \( K \) times, with each fold being used as the test set once.

**Example Code (Python):**

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Define model
model = RandomForestClassifier()

# Perform K-Fold Cross-Validation
scores = cross_val_score(model, X, y, cv=5)  # 5-Fold Cross-Validation

print("Cross-Validation Scores:", scores)
print("Mean Cross-Validation Score:", scores.mean())
```

##### **2.2 Stratified K-Fold Cross-Validation**

Stratified K-Fold ensures that each fold has approximately the same percentage of samples of each class as the original dataset, which is important for imbalanced datasets.

**Example Code (Python):**

```python
from sklearn.model_selection import StratifiedKFold

# Define StratifiedKFold
skf = StratifiedKFold(n_splits=5)

# Perform Stratified K-Fold Cross-Validation
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    print("Test Accuracy:", model.score(X_test, y_test))
```

#### 3. Hyperparameter Tuning

Hyperparameter tuning involves finding the optimal set of parameters for a model to improve its performance. This can be done using techniques like Grid Search and Random Search.

##### **3.1 Grid Search**

Grid Search performs an exhaustive search over a specified parameter grid.

**Example Code (Python):**

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# Define model
model = SVC()

# Perform Grid Search
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X, y)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
```

##### **3.2 Random Search**

Random Search samples a random set of parameters from a specified distribution.

**Example Code (Python):**

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# Define parameter distribution
param_dist = {
    'C': uniform(0.1, 10),
    'kernel': ['linear', 'rbf']
}

# Define model
model = SVC()

# Perform Random Search
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=5)
random_search.fit(X, y)

print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)
```

#### 4. Summary

In this lesson, we covered essential techniques for model evaluation and selection. We discussed various metrics for classification and regression tasks, cross-validation methods to assess model performance, and hyperparameter tuning techniques to optimize models.

In the next lesson, we will explore ensemble learning techniques, which combine multiple models to improve performance and robustness.
