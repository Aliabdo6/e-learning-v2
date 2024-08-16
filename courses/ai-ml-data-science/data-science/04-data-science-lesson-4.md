### **Lesson 4: Introduction to Statistical Analysis**

#### **Objectives:**

1. Understand the basics of statistical analysis.
2. Learn about hypothesis testing and confidence intervals.
3. Apply statistical methods to analyze data.

#### **1. Basics of Statistical Analysis**

Statistical analysis involves collecting, reviewing, and drawing conclusions from data. It helps in understanding data patterns, making predictions, and making informed decisions.

**Key Concepts:**

- **Probability Distributions:** Functions that describe the likelihood of different outcomes.
- **Central Limit Theorem:** States that the distribution of sample means approaches a normal distribution as the sample size increases, regardless of the population distribution.
- **Sampling Methods:** Techniques to select a subset of data from a larger population.

#### **2. Hypothesis Testing**

Hypothesis testing is a statistical method used to make inferences about population parameters based on sample data. The two types of hypotheses are:

- **Null Hypothesis (H0):** Assumes no effect or no difference.
- **Alternative Hypothesis (H1):** Assumes an effect or a difference.

**Steps in Hypothesis Testing:**

1. **Formulate Hypotheses:** State the null and alternative hypotheses.
2. **Select Significance Level (α):** Common choices are 0.05 or 0.01.
3. **Calculate Test Statistic:** Use a statistical test (e.g., t-test) to compute the test statistic.
4. **Determine p-value:** Probability of obtaining results at least as extreme as the observed results.
5. **Make Decision:** Reject or fail to reject the null hypothesis based on the p-value and significance level.

**Code Example: Hypothesis Testing with SciPy**

```python
from scipy import stats

# Sample data
data1 = [23, 25, 30, 35, 40]
data2 = [22, 27, 29, 34, 37]

# Performing t-test
t_statistic, p_value = stats.ttest_ind(data1, data2)

print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")

# Interpreting the result
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis.")
else:
    print("Fail to reject the null hypothesis.")
```

#### **3. Confidence Intervals**

A confidence interval provides a range of values that is likely to contain the population parameter with a certain level of confidence (e.g., 95%).

**Code Example: Calculating Confidence Interval**

```python
import numpy as np
import scipy.stats as stats

# Sample data
data = [23, 25, 30, 35, 40]
mean = np.mean(data)
std_dev = np.std(data, ddof=1)
n = len(data)

# 95% Confidence Interval
confidence = 0.95
z_score = stats.norm.ppf((1 + confidence) / 2)
margin_of_error = z_score * (std_dev / np.sqrt(n))

confidence_interval = (mean - margin_of_error, mean + margin_of_error)
print(f"95% Confidence Interval: {confidence_interval}")
```

#### **4. Applying Statistical Methods**

Use statistical methods to analyze data and make informed decisions. This includes:

- **Correlation Analysis:** Assessing the strength and direction of relationships between variables.
- **Regression Analysis:** Modeling the relationship between a dependent variable and one or more independent variables.

**Code Example: Simple Linear Regression**

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4], [5]])  # Independent variable
y = np.array([1.1, 2.1, 2.9, 4.1, 5.2])  # Dependent variable

# Creating and fitting the model
model = LinearRegression()
model.fit(X, y)

# Making predictions
predictions = model.predict(X)

print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_}")
print(f"Predictions: {predictions}")
```

#### **5. Key Takeaways**

- Statistical analysis helps in making informed decisions based on data.
- Hypothesis testing and confidence intervals are fundamental methods in statistics.
- Applying these methods effectively requires understanding and interpreting the results accurately.

#### **6. Homework/Practice:**

- Perform hypothesis testing on a dataset to test a specific hypothesis.
- Calculate confidence intervals for sample data.
- Apply correlation and regression analysis to explore relationships between variables.
