# -----------------------------
# Mall Customer Segmentation - KMeans (Clean Version)
# -----------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load dataset
data = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

# 2. Select features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# 3. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Elbow Method to find optimal clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # n_init explicitly set
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Method - Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

# 5. KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# 6. Scatter plot of clusters
plt.figure(figsize=(6,5))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], 
            c=data['Cluster'], cmap='Set2', s=60)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments (KMeans)")
plt.show()

# 7. Cluster statistics
for i in range(5):
    print(f"\nCluster {i} stats:")
    print(data[data['Cluster']==i][['Annual Income (k$)', 'Spending Score (1-100)']].describe())
