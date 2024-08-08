import numpy as np

from linear_regression import LinearRegressionBatchGD

# Here we implement the Locally Weighted Regression algorithm
# For the solution of the linear subproblem we use the formula


class LocallyWeightedRegression:
    def __init__(self, tau):
        self.tau = tau

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            y_pred.append(self._predict(X[i]))
        return y_pred

    def _predict(self, x):
        W = self._get_weight(x)
        X = self.X
        y = self.y
        XW = X.T * W
        XWX = XW.dot(X)
        XWX_inv = np.linalg.pinv(XWX)
        XWY = XW.dot(y)
        theta = XWX_inv.dot(XWY)
        return np.dot(x, theta)

    def _get_weight(self, x):
        X = self.X
        W = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            W[i, i] = np.exp(-np.dot(X[i] - x, X[i] - x) / (2 * self.tau**2))
        return W


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X = np.linspace(0, 20, 100)
    y = X**2 + 2 * X + 1 + np.random.normal(0, 1, 100)
    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)
    plt.plot(X, y, "o")
    plt.show()
    topred = np.linspace(0, 20, 5)
    topred = topred.reshape(-1, 1)
    lr = LocallyWeightedRegression(1)
    lr.fit(X, y)
    y_pred = lr.predict(topred)
    plt.plot(X, y, "o")
    plt.plot(topred, y_pred, "ro")
    plt.show()
