import numpy as np
import pandas as pd


class Data:
    def __init__(self, X, y = None, features = None, label= None):
        if X is None:
            raise ValueError("X cannot be None")
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if features is not None and len(X[0]) != len(features):
            raise ValueError("Number of features must match the number of columns in X")
        if features is None:
            features = [f"feat_{str(i)}" for i in range(X.shape[1])]
        if y is not None and label is None:
            label = "y"
        self.X = X
        self.y = y
        self.features = features
        self.label = label


def read_csv(filename,
             sep = ',',
             features = False,
             label = False):
 
    data = pd.read_csv(filename, sep=sep)

    if features and label:
        features = data.columns[:-1]
        label = data.columns[-1]
        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()

    elif features and not label:
        features = data.columns
        X = data.to_numpy()
        y = None

    elif not features and label:
        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()
        features = None
        label = data.columns[-1]

    else:
        X = data.to_numpy()
        y = None
        features = None
        label = None

    return Data(X=X, y=y, features=features, label=label)


def read_parquet(filename):
    data = pd.read_parquet(filename)

    if data.shape[1] != 2:
        raise ValueError("O dataset deve ter exatamente duas colunas: uma independente e uma dependente.")

    features = [f"feat_{i}" for i in range(128)]  # Nomeia as 128 features
    label = data.columns[1]

    X = np.stack(data.iloc[:, 0].to_numpy())  # stack() converte corretamente para (N, 128)
    y = data.iloc[:, 1].to_numpy(dtype=np.float32)

    return Data(X=X, y=y, features=features, label=label)

