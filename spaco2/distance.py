from typing import List

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.neighbors import KDTree

from .logging import logger_manager as lm
from .utils import color_difference_rgb


def spatial_distance(  # TODO: optimize neighbor calculation
    cell_coordinates,
    cell_labels,
    neighbor_weight: float = 0.5,
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

    unique_clusters = np.unique(cell_labels)
    cluster_index = {unique_clusters[i]: i for i in range(len(unique_clusters))}

    # Calculate neighborhoods for all cells
    lm.main_info(f"Calculating cell neighborhood...", indent_level=2)
    tree = KDTree(cell_coordinates, leaf_size=2)
    neighbor_distance_knn, neighbor_index_knn = tree.query(
        cell_coordinates,
        k=n_neighbors,
    )
    # assert len(cell_coordinates) == len(neighbor_distance_knn) == len(neighbor_index_knn)

    # Intersection between knn neighbors and radius neighbors
    lm.main_info(f"Filtering out neighborhood outliers...", indent_level=2)
    neighbor_index_filtered = []
    neighbor_distance_filtered = []
    for i in range(len(cell_coordinates)):
        # filter by radius is equalized to intersection
        neighbor_index_filtered_i = neighbor_index_knn[i][
            neighbor_distance_knn[i] <= radius
        ]
        neighbor_distance_filtered_i = neighbor_distance_knn[i][
            neighbor_distance_knn[i] <= radius
        ]
        # filter banished cell
        if np.sum(cell_labels[neighbor_index_filtered_i] == cell_labels[i]) < n_cells:
            # keep an empty network with only cell i itself
            neighbor_index_filtered_i = np.array([i])
            neighbor_distance_filtered_i = np.array([0])

        neighbor_index_filtered.append(neighbor_index_filtered_i)
        neighbor_distance_filtered.append(neighbor_distance_filtered_i)

    neighbor_index_filtered = np.array(neighbor_index_filtered, dtype=object)
    neighbor_distance_filtered = np.array(neighbor_distance_filtered, dtype=object)

    # Calculate neighbor_matrix
    lm.main_info(f"Calculating cluster neighbor size...", indent_level=2)
    neighbor_matrix = np.zeros(
        [len(unique_clusters), len(unique_clusters)], dtype=np.int32
    )
    for x in range(len(unique_clusters)):
        neighbor_index_cluster_x = neighbor_index_filtered[
            cell_labels == unique_clusters[x]
        ]
        for y in range(len(unique_clusters)):
            if x == y:
                neighbor_matrix[x][y] = 0
            else:
                neighbor_labels_cluster_x = cell_labels[
                    np.unique([j for i in neighbor_index_cluster_x for j in i])
                ]
                neighbor_matrix[x][y] = np.sum(
                    neighbor_labels_cluster_x == unique_clusters[y]
                )
    # Keep maximum between neighbor_matrix[x][y] and neighbor_matrix[y][x]
    for x in range(len(unique_clusters)):
        for y in range(len(unique_clusters)):
            neighbor_matrix[x][y] = max(neighbor_matrix[x][y], neighbor_matrix[y][x])

    # Calculate cluster centroids
    lm.main_info(f"Calculating cluster centroid distance...", indent_level=2)
    cluster_centroids = np.zeros([len(unique_clusters), 2], dtype=np.float64)
    for i in range(len(unique_clusters)):
        cluster_centroids[i] = np.median(
            np.array(cell_coordinates[cell_labels == unique_clusters[i]]), axis=0
        )
    # Calculate cluster centroid distance
    distance_matrix = np.zeros(
        [len(unique_clusters), len(unique_clusters)], dtype=np.float64
    )
    for i in range(len(cluster_centroids)):
        for j in range(i, len(cluster_centroids)):
            distance_matrix[i][j] = distance.euclidean(
                cluster_centroids[i], cluster_centroids[j]
            )
            distance_matrix[j][i] = distance_matrix[i][j]

    # Calculate score matrix
    lm.main_info(f"Calculating cluster interlacement score...", indent_level=2)
    score_matrix = np.zeros(
        [len(unique_clusters), len(unique_clusters)], dtype=np.float64
    )
    for cell_i in range(len(neighbor_index_filtered)):
        size_n_i = len(neighbor_index_filtered[cell_i])
        if size_n_i == 0:
            continue
        cell_cluster_i = cluster_index[cell_labels[cell_i]]
        for j in range(1, size_n_i):
            cell_cluster_j = cluster_index[
                cell_labels[neighbor_index_filtered[cell_i][j]]
            ]
            inversed_euclidean = 1 / neighbor_distance_filtered[cell_i][j]
            score_matrix[cell_cluster_i][cell_cluster_j] += (
                inversed_euclidean / size_n_i
            )
    # Keep maximum between score_matrix[x][y] and score_matrix[y][x], set diagonal to zero
    for x in range(len(unique_clusters)):
        for y in range(len(unique_clusters)):
            if x == y:
                score_matrix[x][y] = 0
            else:
                score_matrix[x][y] = max(score_matrix[x][y], score_matrix[y][x])

    # Merge matrix of different metrics
    lm.main_info(f"Summarizing scores...", indent_level=2)
    """
    # Transform distance_matrix to neighbor_matrix by MinMaxScaler
    mm_transformer = MinMaxScaler(feature_range=(
    np.min(neighbor_matrix), np.max(neighbor_matrix)
    ))
    distance_matrix = mm_transformer.fit_transform(distance_matrix)
    """
    # Transform distance_matrix to neighbor_matrix by Mean
    distance_matrix = (
        -distance_matrix / np.mean(distance_matrix) * np.mean(neighbor_matrix)
    )

    # cluster_interlace_matrix = neighbor_weight * neighbor_matrix + (1 - neighbor_weight) * distance_matrix
    cluster_interlace_matrix = score_matrix

    lm.main_info(f"Constructing cluster interlacement graph...", indent_level=2)
    cluster_interlace_matrix = pd.DataFrame(cluster_interlace_matrix)
    cluster_interlace_matrix.index = unique_clusters
    cluster_interlace_matrix.columns = unique_clusters

    return cluster_interlace_matrix


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
    difference_matrix = np.zeros([len(colors), len(colors)], dtype=np.float32)

    # Calculate difference between colors
    lm.main_info(f"Calculating color perceptual distance...", indent_level=2)
    for i in range(len(colors)):
        for j in range(len(colors)):
            difference_matrix[i][j] = color_difference_rgb(colors[i], colors[j])

    lm.main_info(f"Constructing color distance graph...", indent_level=2)
    difference_matrix = pd.DataFrame(difference_matrix)
    difference_matrix.index = colors
    difference_matrix.columns = colors

    return difference_matrix
