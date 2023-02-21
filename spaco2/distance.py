from typing import List

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from scipy.spatial import distance

from .logging import logger_manager as lm
from .utils import color_difference_rgb


def spatial_distance(  # TODO: optimize neighbor calculation
    cell_coordinates,
    cell_labels,
    cell_weight: float = 0.5, 
    radius: float = 90,  # TODO: comfirm default value
    n_neighbors: int = 4,
    n_cells: int = 3,  # TODO: why n_cells
) -> pd.DataFrame:
    """
    Function to calculate spatial interlacement distance graph for cell clusters, where
    we define the interlacement distance as the number of neighboring cells between two
    cluster.

    Args:
        cell_coordinates: a list like object containing spatial coordinates for each cell.
        cell_labels: a list like object containing cluster labels for each cell.
        cell_weigth: cell weight to calculate cell neighborhood. Defaults to 0.5.
        radius (float, optional): radius used to calculate cell neighborhood. Defaults to 90.
        n_neighbors (int, optional): k for KNN neighbor detection. Defaults to 4.
        n_cells (int, optional): only calculate neighborhood with more than `n_cells`. Defaults to 3.

    Returns:
        pd.DataFrame: a `pandas.DataFrame` with unique cluster names as `index` and `columns`,
            which contains interlacement distance between clusters.
    """

    unique_labels = np.unique(cell_labels)
    distance_matrix = np.zeros([len(unique_labels), len(unique_labels)], dtype=np.int32)

    labels_coordinates = np.zeros([len(unique_labels), 2], dtype=np.int32)
    labels_matrix = np.zeros([len(unique_labels), len(unique_labels)], dtype=np.int32)

    # Calculate cell neighborhood
    lm.main_info(f"Calculating cell neighborhood...", indent_level=2)
    tree = KDTree(cell_coordinates, leaf_size=2)

    for i in range(len(unique_labels)):
        labels_coordinates[i] = np.median(
            np.array(cell_coordinates[cell_labels == unique_labels[i]]), axis=0
        )
        neighbor_index_knn = tree.query(
            cell_coordinates[cell_labels == unique_labels[i]], k=n_neighbors
        )[1]
        neighbor_index_radius = tree.query_radius(
            cell_coordinates[cell_labels == unique_labels[i]], r=radius
        )
        neighbor_index_merged = []
        '''
        # Union between neighbor_index_knn and neighbor_index_radius
        for r in neighbor_index_knn:
            if cell_labels[r][cell_labels == unique_labels[i]].size >= n_cells:
                neighbor_index_merged.append(r)
        for r in neighbor_index_radius:
            if cell_labels[r][cell_labels == unique_labels[i]].size >= n_cells:
                neighbor_index_merged.append(r)
        '''
        # Intersection between neighbor_index_knn and neighbor_index_radius
        for m in range(len(neighbor_index_knn)):
            if (cell_labels[neighbor_index_knn[m]][cell_labels == unique_labels[i]].size
                >= n_cells and 
                cell_labels[neighbor_index_radius[m]][cell_labels == unique_labels[i]].size
                >= n_cells
                ) :
                    neighbor_index_merged.append(
                        np.intersect1d(neighbor_index_knn[m], neighbor_index_radius[m], 
                        assume_unique=True
                    ))

        neighbor_labels = cell_labels[
            np.unique([j for i in neighbor_index_merged for j in i])
        ]
        distance_vector = np.zeros_like(unique_labels)
        for j in range(len(unique_labels)):
            if i == j:
                distance_vector[j] = 0
            else:
                neighbor_count = neighbor_labels[
                    neighbor_labels == unique_labels[j]
                ].size
                if j < i and neighbor_count >= distance_matrix[j][i]:
                    distance_matrix[j][i] = neighbor_count
                elif j < i and neighbor_count < distance_matrix[j][i]:
                    neighbor_count = distance_matrix[j][i]
                distance_vector[j] = neighbor_count
        distance_matrix[i] = distance_vector
        
    # Caculate labels distance
    for i in range(len(labels_coordinates)):
        for j in range(len(labels_coordinates)):
            if i > j:
                labels_matrix[i][j] = labels_matrix[j][i]
            elif i == j:
                labels_matrix[i][j] = 0
            else:
                labels_matrix[i][j] = distance.euclidean(
                labels_coordinates[i], labels_coordinates[j]
                )

    '''
    # Transform labels_matrix to distance_matrix by MinMaxScaler
    mm_transformer = MinMaxScaler(feature_range=(
    np.min(distance_matrix), np.max(distance_matrix)
    ))
    labels_matrix = mm_transformer.fit_transform(labels_matrix)
    '''
    # Transform labels_matrix to distance_matrix by Mean
    labels_matrix = -labels_matrix / np.mean(labels_matrix) * np.mean(distance_matrix)

    distance_matrix = cell_weight * distance_matrix + (1 - cell_weight) * labels_matrix

    lm.main_info(f"Constructing cluster distance graph...", indent_level=2)
    distance_matrix = pd.DataFrame(distance_matrix)
    distance_matrix.index = unique_labels
    distance_matrix.columns = unique_labels

    return distance_matrix


def perceptual_distance(colors: List[str]) -> pd.DataFrame:
    """
    Function to calculate color perceptual difference matrix.
    See `color_difference_rgb` for details.

    Args:
        colors (List[str]): a list of colors (in hex).

    Returns:
        pd.DataFrame: a `pandas.DataFrame` with unique colors (in hex) as `index` and `columns`,
            which contains perceptual distance between colors.
    """
    distance_matrix = np.zeros([len(colors), len(colors)], dtype=np.float32)

    # Calculate difference between colors
    lm.main_info(f"Calculating color perceptual distance...", indent_level=2)
    for i in range(len(colors)):
        for j in range(len(colors)):
            distance_matrix[i][j] = color_difference_rgb(colors[i], colors[j])

    lm.main_info(f"Constructing color distance graph...", indent_level=2)
    distance_matrix = pd.DataFrame(distance_matrix)
    distance_matrix.index = colors
    distance_matrix.columns = colors

    return distance_matrix
