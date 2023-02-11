from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.manifold import MDS
from umap import UMAP

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .logging import logger_manager as lm
from .utils import lab_to_hex, matrix_distance, tsp

# from .tsp import tsp #stochastic tsp solver


def map_graph_tsp(
    cluster_distance: pd.DataFrame,
    color_distance: pd.DataFrame,
    tsp_solver: Literal["exact", "heuristic"] = "heuristic",
    distance_metric: Literal["euclidean", "manhattan", "log"] = "manhattan",
    verbose: bool = False,
) -> Dict[Any, str]:
    """
    Function to map the vertices between two graph based on their longest loop, using
    a traveling salesman problem (TSP) solver.

    Args:
        cluster_distance (pd.DataFrame): a `pandas.DataFrame` with unique cluster names as `index` and `columns`,
            which contains a distance adjacent matrix for clusters, representing the dissimilarity between clusters.
        color_distance (pd.DataFrame): a `pandas.DataFrame` with unique colors (in hex) as `index` and `columns`,
            which contains a distance adjacent matrix for colors, representing the perceptual difference between colors.
        tsp_solver (Literal[&quot;exact&quot;, &quot;heuristic&quot;], optional): tsp solver backend. Defaults to "heuristic".
        distance_metric (Literal[&quot;euclidean&quot;, &quot;manhattan&quot;, &quot;log&quot;], optional): metric used for matrix mapping. Defaults to "manhattan".
        verbose (bool, optional): output info.

    Returns:
        Dict[Any, str]: optimized color mapping for clusters, keys are cluster names, values are hex colors.
    """

    assert (
        cluster_distance.shape == color_distance.shape
    ), "Clusters and colors are not in the same size."

    # Calculate tsp loop for clusters and colors
    lm.main_info(f"Solving TSP for cluster graph...")
    cluster_tsp_path, cluster_tsp_score = tsp(
        distance_matrix=-cluster_distance.to_numpy(), solver_backend=tsp_solver
    )
    lm.main_info(f"Solving TSP for color graph...")
    color_tsp_path, color_tsp_score = tsp(
        distance_matrix=-color_distance.to_numpy(), solver_backend=tsp_solver
    )

    if verbose:
        lm.main_info(f"cluster_tsp_score: {cluster_tsp_score}")
        lm.main_info(f"color_tsp_score: {color_tsp_score}")

    # Reorder cluster distance matrix to match with cluster tsp loop
    cluster_distance_tsp = np.transpose(cluster_distance[cluster_tsp_path])[
        cluster_tsp_path
    ]

    # rotate color tsp loop to find the best match with cluster tsp loop
    lm.main_info(f"Optimizing cluster color mapping...")
    color_tsp_path_rotator = np.concatenate((color_tsp_path, color_tsp_path), axis=0)

    rotate_distance = 1e9
    rotate_offset = 0
    for i in range(len(color_distance)):
        color_rotate_index = color_tsp_path_rotator[i : i + len(color_distance)]
        color_distance_tsp_rotate = np.transpose(color_distance[color_rotate_index])[
            color_rotate_index
        ]
        rotate_distance_tmp = matrix_distance(
            matrix_x=cluster_distance_tsp,
            matrix_y=color_distance_tsp_rotate,
            metric=distance_metric,
        )
        if rotate_distance_tmp <= rotate_distance:
            rotate_distance = rotate_distance_tmp
            rotate_offset = i

    # Return color mapping dictionary, sorted by keys
    color_mapping = dict(
        zip(
            cluster_distance.index[cluster_tsp_path],
            color_distance.index[
                color_tsp_path_rotator[
                    rotate_offset : rotate_offset + len(color_distance)
                ]
            ],
        )
    )
    color_mapping = {k: color_mapping[k] for k in sorted(list(color_mapping.keys()))}

    return color_mapping


def embed_graph(
    cluster_distance: pd.DataFrame,
    transformation: Literal["mds", "umap"] = "umap",
    l_range: Tuple[float, float] = (10, 90),
    log_colors: bool = False,
    trim_fraction: float = 0.0125,
) -> Dict[Any, str]:
    """
    Function to embed the cluster distance graph into chosen colorspace, while keeping distance
    relationship. Currently only supports CIE Lab space. Proper colors are selected within whole
    colorspace based on the embedding of each cluster.

    Args:
        cluster_distance (pd.DataFrame):  a `pandas.DataFrame` with unique cluster names as `index` and `columns`,
            which contains a distance adjacent matrix for clusters, representing the dissimilarity between clusters.
        transformation (Literal[&quot;mds&quot;, &quot;umap&quot;], optional): method used for graph embedding. Defaults to "umap".
        l_range (Tuple[float, float], optional): value range for L channel in LAB colorspace. Defaults to (10,90).
        log_colors (bool, optional): whether to perform log-transformation for color embeddings. Defaults to False.
        trim_fraction (float, optional): quantile for trimming (value clipping). Defaults to 0.0125.

    Returns:
        Dict[Any, str]: auto-generated color mapping for clusters, keys are cluster names, values are hex colors.
    """

    # Embed clusters into 3-dimensional space
    lm.main_info(f"Calculating cluster embedding...")
    if transformation == "mds":
        model = MDS(
            n_components=3,
            dissimilarity="precomputed",
            random_state=123,
        )
    elif transformation == "umap":
        model = UMAP(
            n_components=3,
            metric="precomputed",
            random_state=123,
        )
    embedding = model.fit_transform(cluster_distance)

    # Rescale embedding to CIE Lab colorspace
    lm.main_info(f"Rescaling embedding to CIE Lab colorspace...")
    embedding -= np.quantile(embedding, trim_fraction, axis=0)
    embedding[embedding < 0] = 0
    embedding /= np.quantile(embedding, 1 - trim_fraction)
    embedding[embedding > 1] = 1

    if log_colors:
        embedding = np.log10(embedding + max(np.quantile(embedding, 0.05), 1e-3))
        embedding -= np.min(embedding, axis=0)
        embedding /= np.max(embedding, axis=0)

    embedding[:, 0] *= l_range[1] - l_range[0]
    embedding[:, 0] += l_range[0]
    embedding[:, 1:2] -= 0.5
    embedding[:, 1:2] *= 200

    lm.main_info(f"Generating cluster color mapping...")
    color_mapping = dict(
        zip(
            cluster_distance.index,
            np.apply_along_axis(lab_to_hex, axis=1, arr=embedding),
        )
    )

    return color_mapping


def cluster_mapping_exp(
    adata: AnnData,
    adata_reference: AnnData,
    cluster_key: str,
    mapping_gene_set: List[str] = None,
) -> List:

    assert False, "under development."  # TODO: implement here

    return None


def cluster_mapping_iou(
    cluster_label,
    cluster_label_reference,
) -> List:

    assert False, "under development."  # TODO: implement here

    return None
