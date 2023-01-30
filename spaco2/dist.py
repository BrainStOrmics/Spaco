import numpy as np


def distance_euclidean(X, Y):
    """calculate the euclidean distance between two matrix

    args:
        X: matrix X
        Y: matrix Y
    """
    dist = (X - Y) ** 2
    dist = np.sum(dist)
    dist = np.sqrt(dist)

    return dist


def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly

    args:
        array: a number array
        new_min: the minimum value for new scale
        new_max: the maximum value for new scale
    """
    minimum, maximum = np.min(array), np.max(array)
    if maximum - minimum == 0:
        return array
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum

    return m * array + b


def distance_manhattan(X, Y):
    """calculate the manhattan distance between two matrix

    args:
        X: matrix X
        Y: matrix Y
    """
    return np.sum(abs(X - Y))


def distance_log(X, Y):
    """calculate the log distance between two matrix

    args:
        X: matrix X
        Y: matrix Y
    """
    return np.sum(X * np.log(Y))


def matrix_distance(X, Y, method):
    """calculate the  distance between two matrix

    args:
        X: matrix X
        Y: matrix Y
        method: method for distance
    """
    if len(X) != len(Y) or X.size != Y.size:
        return 1
    if method.lower() == "euclidean":
        return distance_euclidean(X, Y)
    elif method.lower() == "manhattan":
        return distance_manhattan(X, Y)
    elif method.lower() == "log":
        return distance_log(X, Y)
