import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree


def find_neighbors(axis, type, radius, knn, ncell):
    """generate the adjacent matrix for cells

    args:
      axis: axis for each cell
      type: cell type for each cell
      radius: radius for neighbors
      knn: the k-nearest cells
      ncell: the minimum cell number in a subnet for the specific cell type
    """
    tree = KDTree(axis, leaf_size=2)
    keys = np.unique(type)
    vals = np.zeros([len(keys), len(keys)], dtype=np.int32)
    for k in range(len(keys)):
        idx_knn = tree.query(axis[type == keys[k], :], k=knn)[1]
        idx_rds = tree.query_radius(axis[type == keys[k], :], r=radius)
        idx_meg = []
        for r in idx_knn:
            if type[r][type == keys[k]].size >= ncell:
                idx_meg.append(r)
        for r in idx_rds:
            if type[r][type == keys[k]].size >= ncell:
                idx_meg.append(r)
        key = type[np.unique([j for i in idx_knn for j in i])]
        tmp = np.zeros_like(keys)
        for i in range(len(keys)):
            if k == i:
                tmp[i] = 0
            else:
                nbs = key[key == keys[i]].size
                if i < k and nbs >= vals[i][k]:
                    vals[i][k] = nbs
                elif i < k and nbs < vals[i][k]:
                    nbs = vals[i][k]
                tmp[i] = nbs
        vals[k] = tmp

    return keys, vals


def hex_to_rgb(value):
    """convert hex to rgb

    args:
      value: hex value for each color
    """
    value = value.lstrip("#")
    length = len(value)

    return tuple(
        int(value[i : i + length // 3], 16) for i in range(0, length, length // 3)
    )


def color_difference(colorA, colorB):
    """calculate the color difference

    args:
      colorA: color A in hex
      colorB: color B in hex
    """
    rgbA = hex_to_rgb(colorA)
    rgbB = hex_to_rgb(colorB)
    rgbR = (rgbA[0] + rgbB[0]) / 2

    return (
        (2 + rgbR / 256) * (rgbA[0] - rgbB[0]) ** 2
        + 4 * (rgbA[1] - rgbB[1]) ** 2
        + (2 + (255 - rgbR) / 256) * (rgbA[2] - rgbB[2]) ** 2
    ) ** 0.5
    # return ((rgbA[0]-rgbB[0])**2 + (rgbA[1]-rgbB[1])**2 + (rgbA[2]-rgbB[2])**2) ** 0.5


def color_difference_matrix(color):
    """generate the color matrix

    args:
      color: an array for color
    """
    mat = np.zeros([len(color), len(color)], dtype=np.float32)
    for i in range(0, len(color)):
        tmp = np.zeros_like(color, dtype=np.float32)
        for j in range(0, len(color)):
            tmp[j] = color_difference(color[i], color[j])
        mat[i] = tmp

    return mat
