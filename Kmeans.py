import numpy as np

# We define a Kmeans class with appropriate methods


class Kmeans:
    def __init__(self, X, k=3, max_iters=100):
        # X is the data matrix n x d
        self.X = X
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.clusters = None

    def initialize_clusters(self):
        # Randomly initialize the centroids
        self.centroids = self.X[
            np.random.choice(self.X.shape[0], self.k, replace=False)
        ]

    def initialize_clusters_plus_plus(self):
        # Initialize the centroids using the Kmeans++ algorithm
        self.centroids = np.zeros((self.k, self.X.shape[1]))
        self.centroids[0] = self.X[np.random.choice(self.X.shape[0])]
        for i in range(1, self.k):
            dist = np.linalg.norm(self.X[:, None] - self.centroids[:i], axis=2)
            dist = np.min(dist, axis=1)
            probs = dist**2 / np.sum(dist**2)
            self.centroids[i] = self.X[np.random.choice(self.X.shape[0], p=probs)]

    def assign_clusters(self):
        # Assign each data point to the closest centroid
        self.clusters = np.argmin(
            np.linalg.norm(self.X[:, None] - self.centroids, axis=2), axis=1
        )

    def update_centroids(self):
        # Update the centroids based on the mean of the data points
        for i in range(self.k):
            self.centroids[i] = np.mean(self.X[self.clusters == i], axis=0)

    def fit(self):
        # Fit the Kmeans model
        for i in range(self.max_iters):
            old_centroids = self.centroids.copy()
            self.assign_clusters()
            self.update_centroids()
            if np.all(old_centroids == self.centroids):
                break

    def elbow_method(self):
        # Compute the elbow method
        sse = []
        for k in range(1, 11):
            self.k = k
            self.initialize_clusters()
            self.fit()
            sse.append(
                np.sum(
                    np.min(
                        np.linalg.norm(self.X[:, None] - self.centroids, axis=2), axis=1
                    )
                )
            )
        return sse


if __name__ == "__main__":
    # Create a Kmeans object
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=500, n_features=2, centers=3)
    kmeans = Kmeans(X)
    kmeans.initialize_clusters_plus_plus()
    kmeans.fit()
    # Plot the scatter plot
    import matplotlib.pyplot as plt

    plt.scatter(X[:, 0], X[:, 1], c=kmeans.clusters)
    plt.title("Kmeans Clustering")
    plt.show()
    # Also plot the elbow method
    sse = kmeans.elbow_method()
    plt.plot(sse)
    plt.title("Elbow Method")
    plt.show()
