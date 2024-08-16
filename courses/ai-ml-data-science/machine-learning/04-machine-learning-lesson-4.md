### Machine Learning Course - Lesson 4: Unsupervised Learning Algorithms

#### Lesson Overview

In this lesson, we'll dive into unsupervised learning algorithms, which are used for analyzing and interpreting datasets without labeled responses. We will cover clustering, dimensionality reduction, and anomaly detection techniques, with practical code examples.

#### 1. Clustering Algorithms

Clustering algorithms are used to group similar data points together based on their features. We'll explore K-Means and Hierarchical Clustering.

##### **1.1 K-Means Clustering**

K-Means is a popular clustering algorithm that partitions the data into \( k \) distinct clusters based on feature similarity.

**Example Code (Python):**

```python
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, alpha=0.75)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.show()
```

##### **1.2 Hierarchical Clustering**

Hierarchical Clustering creates a hierarchy of clusters by either iteratively merging small clusters into larger ones (agglomerative) or dividing large clusters into smaller ones (divisive).

**Example Code (Python):**

```python
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
iris = load_iris()
X = iris.data

# Apply Agglomerative Clustering
clustering = AgglomerativeClustering(n_clusters=3)
y_clustering = clustering.fit_predict(X)

# Plot results
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_clustering, palette='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Hierarchical Clustering')
plt.show()
```

#### 2. Dimensionality Reduction

Dimensionality Reduction techniques are used to reduce the number of features in a dataset while preserving as much information as possible. We'll cover Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE).

##### **2.1 Principal Component Analysis (PCA)**

PCA reduces the dimensionality of the data by transforming it into a set of orthogonal (uncorrelated) components that capture the maximum variance.

**Example Code (Python):**

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot results
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.show()
```

##### **2.2 t-Distributed Stochastic Neighbor Embedding (t-SNE)**

t-SNE is a non-linear dimensionality reduction technique that is particularly good at preserving local structure in data and is often used for visualizing high-dimensional datasets.

**Example Code (Python):**

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot results
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE of Iris Dataset')
plt.show()
```

#### 3. Anomaly Detection

Anomaly Detection algorithms identify unusual data points that differ significantly from the majority of the data. We'll cover Isolation Forest and Local Outlier Factor (LOF).

##### **3.1 Isolation Forest**

Isolation Forest is an ensemble-based anomaly detection method that isolates anomalies instead of profiling normal data points.

**Example Code (Python):**

```python
from sklearn.ensemble import IsolationForest
import numpy as np

# Sample dataset with anomalies
X = np.array([[0.1, 0.2], [0.2, 0.3], [0.2, 0.2], [10, 10], [11, 11]])

# Apply Isolation Forest
model = IsolationForest(contamination=0.2, random_state=42)
y_pred = model.fit_predict(X)

print("Anomaly Detection Results:", y_pred)
```

##### **3.2 Local Outlier Factor (LOF)**

LOF detects anomalies based on the local density deviation of a data point with respect to its neighbors.

**Example Code (Python):**

```python
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

# Sample dataset with anomalies
X = np.array([[0.1, 0.2], [0.2, 0.3], [0.2, 0.2], [10, 10], [11, 11]])

# Apply Local Outlier Factor
model = LocalOutlierFactor(n_neighbors=2)
y_pred = model.fit_predict(X)

print("Anomaly Detection Results:", y_pred)
```

#### 4. Summary

In this lesson, we covered several unsupervised learning algorithms, including clustering (K-Means and Hierarchical Clustering), dimensionality reduction (PCA and t-SNE), and anomaly detection (Isolation Forest and LOF). These techniques help in exploring, visualizing, and understanding complex datasets.

In the next lesson, we will discuss model evaluation and selection techniques, including metrics for classification and regression, cross-validation, and hyperparameter tuning.
