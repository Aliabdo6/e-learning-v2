### **Lesson 3: Exploratory Data Analysis (EDA)**

#### **Objectives:**

1. Understand the purpose of Exploratory Data Analysis (EDA).
2. Learn techniques for summarizing and visualizing data.
3. Apply EDA methods to uncover insights and patterns.

#### **1. What is Exploratory Data Analysis (EDA)?**

EDA is an approach to analyzing data sets to summarize their main characteristics, often using visual methods. It helps in understanding the data distribution, identifying patterns, detecting anomalies, and testing hypotheses.

**Key Concepts:**

- **Descriptive Statistics:** Basic statistics to summarize data (mean, median, mode, etc.).
- **Data Distribution:** Understanding how data is spread and its central tendencies.
- **Relationships:** Examining correlations and relationships between variables.

#### **2. Descriptive Statistics**

Descriptive statistics provide a summary of data through measures like mean, median, mode, variance, and standard deviation.

**Code Example: Calculating Descriptive Statistics with Pandas**

```python
import pandas as pd

# Sample DataFrame
data = {'Height': [150, 160, 165, 170, 180],
        'Weight': [55, 60, 65, 70, 75]}
df = pd.DataFrame(data)

# Calculating descriptive statistics
print(df.describe())
print(df['Height'].mean())
print(df['Weight'].median())
```

#### **3. Data Visualization**

Visualization helps in understanding data patterns, trends, and distributions. Common plots include:

- **Histograms:** Show the distribution of a single variable.
- **Box Plots:** Display data distribution and identify outliers.
- **Scatter Plots:** Examine relationships between two variables.
- **Correlation Matrix:** Visualize correlations between multiple variables.

**Code Example: Visualization with Matplotlib and Seaborn**

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Sample DataFrame
df = pd.DataFrame({'Height': [150, 160, 165, 170, 180],
                   'Weight': [55, 60, 65, 70, 75]})

# Histogram
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(df['Height'], bins=5, edgecolor='black')
plt.title('Height Distribution')

# Scatter Plot
plt.subplot(1, 2, 2)
plt.scatter(df['Height'], df['Weight'])
plt.title('Height vs Weight')
plt.xlabel('Height')
plt.ylabel('Weight')

plt.tight_layout()
plt.show()

# Box Plot using Seaborn
sns.boxplot(x='Height', data=df)
plt.title('Box Plot of Height')
plt.show()
```

#### **4. Identifying Patterns and Anomalies**

Use visual and statistical methods to find patterns, correlations, or anomalies in the data.

**Code Example: Correlation Matrix**

```python
# Sample DataFrame with more variables
df = pd.DataFrame({'Height': [150, 160, 165, 170, 180],
                   'Weight': [55, 60, 65, 70, 75],
                   'Age': [23, 25, 30, 35, 40]})

# Calculating correlation matrix
corr_matrix = df.corr()
print(corr_matrix)

# Plotting correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
```

#### **5. Key Takeaways**

- EDA helps in summarizing and visualizing data to uncover insights.
- Use descriptive statistics to understand data characteristics.
- Employ various visualization techniques to explore data relationships and distributions.

#### **6. Homework/Practice:**

- Load a dataset and perform basic descriptive statistics.
- Create various plots (histogram, box plot, scatter plot) to visualize the data.
- Analyze the visualizations to identify patterns or anomalies in the dataset.
