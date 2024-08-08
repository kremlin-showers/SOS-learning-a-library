import numpy as np
from tqdm import trange

import preprocessing


class LinearRegressionBatchGD:
    def __init__(
        self, learning_rate=0.01, epochs=100, batch_size=1, weights=None, verbose=False
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = weights
        self.verbose = verbose
        self.errors = None

    def rmse_loss(self, X, y, theta):
        # X has shape nxd
        # Weights will have shape dx1
        # y has shape nx1
        assert X.shape[0] == y.shape[0]
        y_pred = np.dot(X, theta)
        return np.sqrt(np.mean((y - y_pred) ** 2))

    def grad(self, X, y, theta):
        # We will return the gradient with shape dx1
        # X has shape nxd
        # Weights will have shape dx1
        # y has shape nx1
        grad = (np.dot(X.T, np.dot(X, theta) - y)) / X.shape[0]
        return grad

    def create_batches(self, X, y, batch_size, n):
        batches = []
        data = np.hstack((X, y))
        np.random.shuffle(data)
        batch_num = n // batch_size
        for i in range(batch_num):
            X_batch = data[i * batch_size : (i + 1) * batch_size, :-1]
            y_batch = data[i * batch_size : (i + 1) * batch_size, -1]
            batches.append((X_batch, y_batch))
            if i == batch_num - 1:
                X_batch = data[i * batch_size :, :-1]
                y_batch = data[i * batch_size :, -1]
                batches.append((X_batch, y_batch))
        return batches

    def fit(self, X, y):
        self.errors = []
        n, d = X.shape
        if self.weights is None:
            self.weights = np.zeros((d, 1))

        batches = self.create_batches(X, y, self.batch_size, n)

        for i in range(self.epochs):
            J_0 = self.rmse_loss(X, y, self.weights)
            self.errors.append(J_0)
            for batch in batches:
                X_batch, y_batch = batch
                current_weights = self.weights.copy()
                self.weights -= self.learning_rate * self.grad(
                    X_batch, y_batch, self.weights
                )
                J_1 = self.rmse_loss(X, y, self.weights)
                if J_1 - J_0 > 1e-6:
                    self.weights = current_weights
                    self.learning_rate /= 2
                else:
                    J_0 = J_1

            if self.verbose:
                print(f"Epoch: {i}, Loss: {J_0}")

        print("Training Complete")
        print("Final Loss: ", self.rmse_loss(X, y, self.weights))
        print("Final Weights: ", self.weights)
        return self.weights, self.errors

    def close_form_solution(self, X, y):
        return np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y))

    def predict(self, X, weights=None):
        if weights is not None:
            return np.dot(X, weights)
        return np.dot(X, self.weights)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd

    import preprocessing

    dataset = pd.read_csv(
        "./Datasets/Gradient_descent.txt", sep=" ", header=None, names=["x", "y"]
    )
    X = dataset[["x"]].values
    X = preprocessing.add_bias(X)
    y = dataset["y"].values.reshape(-1, 1)
    lr = LinearRegressionBatchGD(learning_rate=0.2, epochs=5000, batch_size=1)
    lr.fit(X, y)

    plt.scatter(X[:, 1], y, color="red")
    plt.plot(X[:, 1], lr.predict(X), color="blue", label="Gradient Descent")
    plt.legend()
    plt.title("Batch Gradient Descent")
    plt.show()

    # Plot the errors too
    plt.plot(range(len(lr.errors)), lr.errors)
    plt.title("Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    # Plot the analytical solution
    weights = lr.close_form_solution(X, y)
    print(lr.rmse_loss(X, y, weights))

    plt.scatter(X[:, 1], y, color="red")
    plt.plot(X[:, 1], lr.predict(X, weights), color="blue", label="Analytical Solution")
    plt.title("The Analytical Solution")
    plt.show()
