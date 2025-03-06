import numpy as np


def accuracy(y_true, y_pred):

    def correct_format(y):
        if len(y[0]) == 1:
            corrected_y = [np.round(y[i][0]) for i in range(len(y))]
        else:
            corrected_y = [np.argmax(y[i]) for i in range(len(y))]
        return np.array(corrected_y)
    if isinstance(y_true[0], list) or isinstance(y_true[0], np.ndarray):
        y_true = correct_format(y_true)
    if isinstance(y_pred[0], list) or isinstance(y_pred[0], np.ndarray):
        y_pred = correct_format(y_pred)
    return np.sum(y_pred == y_true) / len(y_true)


def mse(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2) / len(y_true)


def mse_derivative(y_true, y_pred):
    return 2 * np.sum(y_true - y_pred) / len(y_true)

def binary_crossentropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15) 
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_crossentropy_derivative(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return (y_pred - y_true) / (y_pred * (1 - y_pred)) / len(y_true)