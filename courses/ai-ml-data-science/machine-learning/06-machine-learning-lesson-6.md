### Machine Learning Course - Lesson 6: Ensemble Learning

#### Lesson Overview

In this lesson, we will explore ensemble learning techniques, which combine the predictions from multiple models to improve overall performance. Ensemble methods can enhance the accuracy, robustness, and generalization of machine learning models. We will cover three main ensemble techniques: Bagging, Boosting, and Stacking.

#### 1. Bagging (Bootstrap Aggregating)

Bagging is a technique where multiple models are trained on different subsets of the training data and their predictions are averaged (for regression) or voted upon (for classification). The idea is to reduce variance and prevent overfitting.

##### **1.1 Bagging with Decision Trees**

**Example Code (Python):**

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define base model
base_model = DecisionTreeClassifier()

# Define Bagging model
bagging_model = BaggingClassifier(base_model, n_estimators=50, random_state=42)
bagging_model.fit(X_train, y_train)

# Make predictions
y_pred = bagging_model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
```

#### 2. Boosting

Boosting is a technique that sequentially trains models, where each new model attempts to correct the errors of the previous models. This method focuses on the instances that were misclassified by previous models.

##### **2.1 Boosting with Gradient Boosting Machines (GBM)**

**Example Code (Python):**

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define Gradient Boosting model
gbm_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbm_model.fit(X_train, y_train)

# Make predictions
y_pred = gbm_model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
```

##### **2.2 Boosting with XGBoost**

XGBoost (Extreme Gradient Boosting) is a popular and efficient implementation of gradient boosting.

**Example Code (Python):**

```python
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define XGBoost model
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
```

#### 3. Stacking

Stacking involves training multiple models (base learners) and combining their predictions using another model (meta-learner). This method leverages the strengths of various models to improve performance.

##### **3.1 Stacking with Logistic Regression as Meta-Learner**

**Example Code (Python):**

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define base learners
base_learners = [
    ('dt', DecisionTreeClassifier()),
    ('svc', SVC(probability=True))
]

# Define meta-learner
meta_learner = LogisticRegression()

# Define Stacking model
stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_learner)
stacking_model.fit(X_train, y_train)

# Make predictions
y_pred = stacking_model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
```

#### 4. Summary

In this lesson, we covered ensemble learning techniques including Bagging, Boosting, and Stacking. These methods help to improve model performance by combining multiple models in different ways.

- **Bagging** reduces variance and prevents overfitting by averaging multiple models.
- **Boosting** improves model performance by focusing on errors from previous models.
- **Stacking** combines predictions from various models to leverage their strengths.

In the next lesson, we will explore advanced topics in machine learning, including deep learning, neural networks, and techniques for handling large-scale datasets.
