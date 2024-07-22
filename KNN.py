import numpy as np
from scipy.stats import mode

import preprocessing

# We implement here the KNN algorithm.


def KNN_Classifier(X_train, y_train, X_test, k=3):
    """
    K-Nearest Neighbors Classifier
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data
    :param k: Number of neighbors
    :return: Predicted labels
    """
    dists = np.zeros((X_train.shape[0], X_test.shape[0]))
    dists = np.sqrt(
        (X_train**2).sum(axis=1)[:, np.newaxis]
        + (X_test**2).sum(axis=1)
        - 2 * X_train.dot(X_test.T)
    )
    print(dists)
    nearest_dists = np.argsort(dists, axis=0)
    print(nearest_dists)
    nearest_labels = y_train[nearest_dists]
    print(nearest_labels[:k, :])
    modes = mode(nearest_labels[:k, :])
    return modes.mode.ravel()


def KNN_Regression(X_train, y_train, X_test, k=3):
    """
    K-Nearest Neighbors Regressor
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data
    :param k: Number of neighbors
    :return: Predicted labels
    """
    dists = np.zeros((X_train.shape[0], X_test.shape[0]))
    dists = np.sqrt(
        (X_train**2).sum(axis=1)[:, np.newaxis]
        + (X_test**2).sum(axis=1)
        - 2 * X_train.dot(X_test.T)
    )
    nearest_dists = np.argsort(dists, axis=0)
    nearest_labels = y_train[nearest_dists]
    return np.mean(nearest_labels[:k, :], axis=0)


if __name__ == "__main__":
    # Just demonstrating KNN on the breast cancer dataset
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import sklearn.model_selection

    df = pd.read_csv("./Datasets/breast-cancer-wisconsin.data.txt", header=None)
    col_names = [
        "Id",
        "Clump_thickness",
        "Uniformity_Cell_Size",
        "Uniformity_Cell_Shape",
        "Marginal_Adhesion",
        "Single_Epithelial_Cell_Size",
        "Bare_Nuclei",
        "Bland_Chromatin",
        "Normal_Nucleoli",
        "Mitoses",
        "Class",
    ]
    df.columns = col_names
    df.drop("Id", axis=1, inplace=True)
    df["Bare_Nuclei"] = pd.to_numeric(df["Bare_Nuclei"], errors="coerce")
    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    for df1 in [X_train, X_test]:
        for col in X_train.columns:
            col_median = X_train[col].median()
            df1[col].fillna(col_median, inplace=True)

    X_train, X_test, y_train, y_test = preprocessing.train_test_split(
        X_train, y_train, test_size=0.2
    )

    X_train, X_train_means, X_train_stds = preprocessing.Normalize_X(X_train)
    X_test = (X_test - X_train_means) / X_train_stds
    y_pred = KNN_Classifier(X_train, y_train, X_test, k=3)
    print(preprocessing.accuracy(y_test, y_pred))
