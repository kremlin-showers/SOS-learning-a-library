import matplotlib.pyplot as plt
import numpy as np


def Normalize_X(X):
    # Normalizes the input data X (nxd matrix) to have zero mean and unit variance
    # Returns the normalized data X, the mean of each column, and the standard deviation of each column
    x_means = np.mean(X, axis=0)
    x_stds = np.std(X, axis=0)
    X = (X - x_means) / x_stds
    return X, x_means, x_stds


def Normalize_min(X):
    # Normalises using minmax scaler
    x_max = np.max(X, axis=0)
    x_min = np.min(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    return X, x_max, x_min


def add_bias(X):
    # Adds a bias column to the input data X (nxd matrix)
    # Returns a new nx(d+1) matrix with the bias column of all ones added
    n, d = X.shape
    bias = np.ones((n, 1))
    X = np.hstack((bias, X))
    return X


def accuracy(y_true, y_pred):
    # Calculates the accuracy of the predicted labels y_pred (nx1 vector) with respect to the true labels y_true (nx1 vector)
    # Returns the accuracy as a percentage
    # For Clssification problems
    return np.mean(y_true == y_pred) * 100


def train_test_split(X, y, test_size=0.25):
    if isinstance(test_size, float):
        test_size = round(test_size * len(X))

    data = np.column_stack((X, y))
    np.random.shuffle(data)
    X = data[:, :-1]
    y = data[:, -1]
    X_test = X[:test_size]
    y_test = y[:test_size]
    X_train = X[test_size:]
    y_train = y[test_size:]
    return X_train, X_test, y_train, y_test


class MinMaxScaler:
    def __init__(self):
        self.x_max = None
        self.x_min = None

    def fit(self, X, y=None):
        # fits the scaler to the data
        self.x_max = np.max(X, axis=0)
        self.x_min = np.min(X, axis=0)

    def transform(self, X):
        # Scales
        return (X - self.x_min) / (self.x_max - self.x_min)

    def fit_transform(X, y=None):
        # Fits and scales
        self.fit(X)
        return self.transform(X)


def create_batches(X, y, batch_size):
    """
    This function is used to create the batches of randomly selected data points.

    Args:
      X : 2D numpy array of data points. Dimensions (n x (d+1))
      y : 2D numpy array of target values. Dimensions (n x 1)

    Returns:
      batches : list of tuples with each tuple of size batch size.
    """
    batches = []
    data = np.hstack((X, y))
    np.random.shuffle(data)
    num_batches = data.shape[0] // batch_size
    i = 0
    for i in range(num_batches + 1):
        if i < num_batches:
            batch = data[i * batch_size : (i + 1) * batch_size, :]
            X_batch = batch[:, :-1]
            Y_batch = batch[:, -1].reshape((-1, 1))
            batches.append((X_batch, Y_batch))
        if data.shape[0] % batch_size != 0 and i == num_batches:
            batch = data[i * batch_size : data.shape[0]]
            X_batch = batch[:, :-1]
            Y_batch = batch[:, -1].reshape((-1, 1))
            batches.append((X_batch, Y_batch))
    return batches


class StandardScaler:
    def __init__(self):
        self.x_means = None
        self.x_stds = None

    def fit(self, X, y=None):
        self.x_means = np.mean(X, axis=0)
        self.x_stds = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.x_means) / self.x_stds

    def fit_transform(X, y=None):
        self.fit(X)
        return self.transform(X)


def plot_loss(error_list, batch_size):
    """
    This function plots the loss for each epoch.

    Args:
      error_list : list of validation loss for each epoch
      batch_size : size of one batch
    Returns:
      None
    """
    # Complete this function to plot the graph of losses stored in model's "error_list"
    # Save the plot in "figures" folder.
    plt.plot(error_list)
    plt.title("Loss vs Epochs for batch size: " + str(batch_size))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
