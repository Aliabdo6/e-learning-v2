### Machine Learning Course - Lesson 2: Data Preparation and Preprocessing

#### Lesson Overview

In this lesson, we'll dive into the critical steps of data preparation and preprocessing. These steps are essential for ensuring that the data is clean, relevant, and ready for use in training machine learning models. We'll cover data cleaning, feature engineering, and data splitting techniques, with hands-on examples to illustrate each concept.

#### 1. Data Cleaning

Data cleaning involves identifying and correcting errors or inconsistencies in the dataset. This step is crucial as noisy or incomplete data can significantly impact the performance of your ML models.

**Common Data Cleaning Tasks:**

- **Handling Missing Values:** Missing data can be handled by removing rows with missing values, imputing with statistical measures, or using algorithms that can handle missing values.
- **Removing Duplicates:** Duplicate entries can skew the results and need to be removed.
- **Outlier Detection:** Outliers are data points that differ significantly from other observations. They can be detected using statistical methods or visualizations and addressed accordingly.

**Example Code (Python):**

```python
import pandas as pd
import numpy as np

# Load dataset
data = pd.DataFrame({
    'Feature1': [1, 2, np.nan, 4, 5, 5, 7],
    'Feature2': [8, 8, 8, 8, np.nan, 10, 10],
    'Feature3': [1, 2, 3, 4, 5, 6, 100]  # Outlier in Feature3
})

# Handling missing values
data['Feature1'].fillna(data['Feature1'].mean(), inplace=True)  # Impute with mean
data['Feature2'].dropna(inplace=True)  # Drop rows with missing values

# Removing duplicates
data = data.drop_duplicates()

# Outlier detection and removal (example using Z-score)
from scipy import stats
z_scores = np.abs(stats.zscore(data[['Feature1', 'Feature2', 'Feature3']]))
data = data[(z_scores < 3).all(axis=1)]

print(data)
```

#### 2. Feature Engineering

Feature engineering involves creating new features or modifying existing ones to improve the performance of machine learning models. This can include normalization, encoding categorical variables, and creating interaction features.

**Common Feature Engineering Techniques:**

- **Normalization/Standardization:** Scaling features to a common range or distribution to ensure that they contribute equally to the model.
- **One-Hot Encoding:** Converting categorical variables into binary vectors to be used in ML algorithms.
- **Feature Interaction:** Creating new features that represent interactions between existing features.

**Example Code (Python):**

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sample dataset
data = pd.DataFrame({
    'Age': [25, 32, 45, 23, 35],
    'Income': [50000, 60000, 75000, 40000, 65000],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female']
})

# Normalization
scaler = StandardScaler()
data[['Age', 'Income']] = scaler.fit_transform(data[['Age', 'Income']])

# One-Hot Encoding
encoder = OneHotEncoder(drop='first')
encoded_gender = encoder.fit_transform(data[['Gender']]).toarray()
encoded_gender_df = pd.DataFrame(encoded_gender, columns=encoder.get_feature_names_out(['Gender']))

# Combine with original data
data = pd.concat([data.drop('Gender', axis=1), encoded_gender_df], axis=1)
print(data)
```

#### 3. Data Splitting

Data splitting involves dividing the dataset into training and testing sets. This is essential for evaluating the performance of your ML models and ensuring that they generalize well to new, unseen data.

**Common Data Splitting Techniques:**

- **Train-Test Split:** Dividing the data into a training set and a test set.
- **Cross-Validation:** Splitting the data into multiple folds and using each fold for testing while training on the remaining folds to get a more robust evaluation.

**Example Code (Python):**

```python
from sklearn.model_selection import train_test_split

# Sample dataset
X = data[['Age', 'Income', 'Gender_Female']]
y = np.random.choice([0, 1], size=len(data))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set:", X_train.shape, y_train.shape)
print("Testing set:", X_test.shape, y_test.shape)
```

#### 4. Summary

In this lesson, we explored the essential steps in data preparation and preprocessing. We covered data cleaning to handle missing values and outliers, feature engineering to create and transform features, and data splitting to prepare the data for model training and evaluation.

In the next lesson, we will delve into various machine learning algorithms, including both supervised and unsupervised methods, and see how to apply them to real-world problems.
