### **Lesson 7: Unsupervised Learning – Clustering and Dimensionality Reduction**

#### **Objectives:**

1. Understand the principles of unsupervised learning.
2. Learn about clustering algorithms and their applications.
3. Explore dimensionality reduction techniques to simplify data.

---

#### **1. Unsupervised Learning Overview**

Unsupervised learning involves training models on data without labeled responses. The goal is to uncover hidden patterns or structures in the data.

**Types:**

- **Clustering:** Grouping similar data points together based on features.
- **Dimensionality Reduction:** Reducing the number of features while preserving important information.

---

#### **2. Clustering Techniques**

Clustering algorithms group data points into clusters based on similarity. Common clustering algorithms include:

**a. K-Means Clustering**
K-Means partitions data into K clusters by minimizing the variance within each cluster.

**Code Example: K-Means Clustering**

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Create and fit model
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# Predict cluster labels
y_kmeans = kmeans.predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

**b. Hierarchical Clustering**
Hierarchical clustering creates a tree of clusters, where each node represents a cluster.

**Code Example: Hierarchical Clustering**

```python
import scipy.cluster.hierarchy as sch

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Perform hierarchical clustering
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))

plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()
```

**c. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
DBSCAN identifies clusters based on density and can find clusters of arbitrary shape.

**Code Example: DBSCAN**

```python
from sklearn.cluster import DBSCAN

# Create and fit model
dbscan = DBSCAN(eps=0.3, min_samples=10)
y_dbscan = dbscan.fit_predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_dbscan, s=50, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

---

#### **3. Dimensionality Reduction Techniques**

Dimensionality reduction reduces the number of features while retaining as much information as possible. Common techniques include:

**a. Principal Component Analysis (PCA)**
PCA transforms data into a new coordinate system such that the greatest variance lies on the first coordinate (principal component).

**Code Example: PCA**

```python
from sklearn.decomposition import PCA

# Fit PCA model
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plotting
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', s=50)
plt.title('PCA Result')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
```

**b. t-Distributed Stochastic Neighbor Embedding (t-SNE)**
t-SNE reduces dimensionality while preserving the local structure of the data, often used for visualization.

**Code Example: t-SNE**

```python
from sklearn.manifold import TSNE

# Fit t-SNE model
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X)

# Plotting
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c='green', s=50)
plt.title('t-SNE Result')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()
```

---

#### **4. Key Metrics and Evaluation**

- **For Clustering:**
  - **Silhouette Score:** Measures how similar an object is to its own cluster compared to other clusters.
  - **Davies-Bouldin Index:** Measures the average similarity ratio of each cluster with the one that is most similar to it.

**Code Example: Silhouette Score**

```python
from sklearn.metrics import silhouette_score

# Calculate silhouette score
silhouette_avg = silhouette_score(X, y_kmeans)
print(f"Silhouette Score: {silhouette_avg}")
```

- **For Dimensionality Reduction:**
  - **Explained Variance Ratio:** Proportion of variance explained by each principal component.

**Code Example: Explained Variance Ratio**

```python
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
```

---

#### **5. Key Takeaways**

- **Clustering** helps group similar data points and understand data structure.
- **Dimensionality Reduction** simplifies data and helps in visualization and reducing computational costs.
- Proper evaluation and understanding of techniques are crucial for effective data analysis.

#### **6. Homework/Practice:**

- Apply clustering algorithms to a dataset and evaluate the results using metrics like silhouette score.
- Use dimensionality reduction techniques to visualize high-dimensional data.
- Experiment with different parameter settings for clustering and dimensionality reduction algorithms.
