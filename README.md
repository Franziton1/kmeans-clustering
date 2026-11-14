\# K-Means Unsupervised Clustering Pipeline



An end-to-end unsupervised machine learning clustering pipeline in Python featuring feature standardization, optimal centroid initialization (`k-means++`), and cluster visualization using Scikit-learn.



\## 📊 Cluster Visualization

!\[K-Means Plot](kmeans\_plot.png)



\## 🚀 Project Overview

\* \*\*Dataset:\*\* 200 data points across 2 numerical features (`X1`, `X2`) with no missing values.

\* \*\*Preprocessing:\*\* Feature centering and standardization (`StandardScaler`) to ensure balanced distance weighting during cluster assignment.

\* \*\*Model Configuration:\*\* `KMeans` with $k=4$ clusters, `k-means++` centroid initialization, and optimized maximum iterations.



\## 📂 Repository Contents

\* `kmeans.py`: Core Python script executing scaling, clustering, and plot generation.

\* `data\_kmeans.csv`: Input dataset.

\* `kmeans\_plot.png`: Visual scatter plot displaying the 4 distinct clusters and their computed centroids.



\## 🛠️ Requirements \& Libraries

Built using Python with:

\* `scikit-learn`

\* `pandas`

\* `numpy`

\* `matplotlib`

