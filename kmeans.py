# Import libraries
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Check library versions
print('sklearn: %s' % sklearn.__version__)
print('numpy: %s' % np.__version__)
print('pandas: %s' % pd.__version__)

# 1. Load Data
df = pd.read_csv('data_kmeans.csv')
print("\nDataset Info:")
df.info()

# 2. Feature Extraction & Standardization
X = df.iloc[:, :].values
sc_X = StandardScaler()
X_scaled = sc_X.fit_transform(X)
print('\nFeatures after scaling (first 5 rows):\n', X_scaled[:5])

# 3. Build and Fit K-Means Model (k=4)
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)
print('\nCluster assignments generated successfully!')

# 4. Plot and Save Clusters with Centroids
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[y_kmeans == 0, 0], X_scaled[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X_scaled[y_kmeans == 1, 0], X_scaled[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X_scaled[y_kmeans == 2, 0], X_scaled[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X_scaled[y_kmeans == 3, 0], X_scaled[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')

# Plot Centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', marker='o', edgecolor='black', label='Centroids')

plt.title('K-Means Clustering (k=4)')
plt.xlabel('Standardized X1')
plt.ylabel('Standardized X2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Save plot image for GitHub & README
plt.savefig('kmeans_plot.png', dpi=300, bbox_inches='tight')
plt.show()