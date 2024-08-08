import numpy as np


class LogisticRegressor:
    def __init__(
        self,
        epochs=1000,
        learning_rate=0.01,
        lambda_value=0,
        tol=1e-4,
        regularisation="l2",
    ):
        # Epochs is the number of iterations
        # Learning rate is the step size
        # Lambda value is the reularisation parameter
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lambda_value = lambda_value
        self.regularisation = regularisation
        assert self.regularisation in [
            "l1",
            "l2",
            "none",
        ], "Regularisation should be either l1, l2 or none"
        self.tolerence = tol

    def sigmoid(self, z):
        """
        function to compute sigmoid of a vector z

        args:    z ---> nx1
        returns: nx1
        """
        z = np.array(z)  # z ---> n x 1
        return 1 / (1 + np.exp(-z))

    def loss_grad_function(self, X, y, theta, m):
        """
        function to compute the loss and gradient

        args:    X ---> nxd, y ---> nx1, theta ---> dx1, m ---> n
        returns: J ---> 1x1, grad ---> 1xd
        """
        """
            The vector a has dimensions nx1
            The vector b has dimensions nx1
            c is a scalar
        """
        a = self.sigmoid(np.dot(X, theta))
        J = -np.sum(y * np.log(a) + (1 - y) * np.log(1 - a)) / m
        grad = np.dot((a - y).T, X) / m
        if self.lambda_value != 0 and self.regularisation == "l2":
            # Do not take bias term into account for regularization
            J += self.lambda_value * np.sum(theta[1:] ** 2) / m
            grad += 2 * self.lambda_value * theta.T[:, 1:] / m
        elif self.lambda_value != 0 and self.regularisation == "l1":
            J += self.lambda_value * np.sum(np.abs(theta[1:])) / m
            grad += self.lambda_value * np.sign(theta.T[:, 1:]) / m
        return (J, grad)

    def gradient_descent(self, X, y, theta, m):
        """
        function to implement gradient descent

        args:    X ---> nxd, y ---> nx1, theta ---> dx1, m ---> n
        returns: theta ---> dx1
        """
        ## fixed number of training epochs in self.epochs by making calls to loss_grad_function
        ## Print the losses returned by loss_grad_function over
        ## epochs to keep track of whether gradient descent is converging
        for i in range(self.epochs):
            J = self.loss_grad_function(X, y, theta, m)[0]
            grad = self.loss_grad_function(X, y, theta, m)[1]
            if i % 100 == 0:
                print(f"Loss in epoch {i} is {J}")
            current_theta = theta.copy()
            theta -= self.learning_rate * grad.T
            new_J = self.loss_grad_function(X, y, theta, m)[0]
            if new_J - J > 10e-5:
                theta = current_theta
                self.learning_rate /= 2
            if 1e-8 > J - new_J > 0:
                self.learning_rate *= 1.05

            if np.max(np.abs(grad)) < self.tolerence:
                print("Stopping execution at epoch ", i)
                break
        return theta

    def predict(self, X, theta, probability=False):
        """
        function to predict the class labels

        args:    X ---> nxd, theta ---> dx1
        returns: nx1
        """
        if probability:
            return self.sigmoid(np.dot(X, theta))
        else:
            return np.round(self.sigmoid(np.dot(X, theta)))


if __name__ == "__main__":
    # Demonstrating the use of the LogisticRegressor class
    # First we need to create a dataset
    # Set seed for random
    np.random.seed(42)
    X = []
    y = []
    for i in range(50):
        x, x_1 = np.random.multivariate_normal([3, 2], [[1, 0], [0, 1]])
        X.append([x, x_1])
        y.append(0)
    for i in range(50):
        x, x_1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]])
        X.append([x, x_1])
        y.append(1)

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    m, n = X.shape
    theta = np.random.randn(n, 1)
    log_reg = LogisticRegressor(
        epochs=5000, learning_rate=0.05, lambda_value=0.1, regularisation="none"
    )
    theta = log_reg.gradient_descent(X, y, theta, m)
    print(theta)
    import matplotlib.pyplot as plt

    plt.scatter(X[:, 1], X[:, 2], c=y)
    plt.title("Scatter plot of the data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    # We also plot the line of seperation
    x_values = np.linspace(-2, 5, 100)
    y_values = -(theta[0] + theta[1] * x_values) / theta[2]
    plt.plot(x_values, y_values, label="Decision Boundary")
    plt.legend()
    plt.show()
