import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("data/Mall_Customers.csv")

# Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# K-Means Model
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42,
    n_init=10
)

# Fit and Predict
y_kmeans = kmeans.fit_predict(X)

# Plot Clusters
plt.figure(figsize=(10, 7))

plt.scatter(
    X.iloc[y_kmeans == 0, 0],
    X.iloc[y_kmeans == 0, 1],
    label='Cluster 1'
)

plt.scatter(
    X.iloc[y_kmeans == 1, 0],
    X.iloc[y_kmeans == 1, 1],
    label='Cluster 2'
)

plt.scatter(
    X.iloc[y_kmeans == 2, 0],
    X.iloc[y_kmeans == 2, 1],
    label='Cluster 3'
)

plt.scatter(
    X.iloc[y_kmeans == 3, 0],
    X.iloc[y_kmeans == 3, 1],
    label='Cluster 4'
)

plt.scatter(
    X.iloc[y_kmeans == 4, 0],
    X.iloc[y_kmeans == 4, 1],
    label='Cluster 5'
)

# Centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    marker='X',
    label='Centroids'
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()

plt.savefig("customer_segments.png")
plt.show()

# Add cluster labels to dataset
df["Cluster"] = y_kmeans

print("\nCustomers per Cluster:")
print(df["Cluster"].value_counts().sort_index())

# Save clustered dataset
df.to_csv("customer_segments.csv", index=False)

print("\nClustered dataset saved as customer_segments.csv")

print("Customer Segmentation Completed!")