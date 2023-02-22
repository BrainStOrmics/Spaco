from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .logging import logger_manager as lm
from .utils import lab_to_hex, matrix_distance, tsp


def map_graph_tsp(
    cluster_distance: pd.DataFrame,
    color_distance: pd.DataFrame,
    tsp_solver: Literal["exact", "heuristic"] = "heuristic",
    reproducible: bool = True,
    random_seed: int = 123,
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
        reproducible (bool): whether to guarantee reproducibility when using heuristic solver, if `True`,
            heuristic result will be reproducible under same `random_seed`. If `False`, user can roll different
            results and select their own optimal.
        random_seed (int): random seed for heuristic solver.
        distance_metric (Literal[&quot;euclidean&quot;, &quot;manhattan&quot;, &quot;log&quot;], optional): metric used for matrix mapping. Defaults to "manhattan".
        verbose (bool, optional): output info.

    Returns:
        Dict[Any, str]: optimized color mapping for clusters, keys are cluster names, values are hex colors.
    """

    assert (
        cluster_distance.shape == color_distance.shape
    ), "Clusters and colors are not in the same size."

    # Calculate tsp loop for clusters and colors
    lm.main_info(f"Solving TSP for cluster graph...", indent_level=2)
    cluster_tsp_path, cluster_tsp_score = tsp(
        distance_matrix=-cluster_distance.to_numpy(),
        solver_backend=tsp_solver,
        reproducible=reproducible,
        random_seed=random_seed,
    )
    lm.main_info(f"Solving TSP for color graph...", indent_level=2)
    color_tsp_path, color_tsp_score = tsp(
        distance_matrix=-color_distance.to_numpy(),
        solver_backend=tsp_solver,
        reproducible=reproducible,
        random_seed=random_seed,
    )

    if verbose:
        lm.main_info(f"cluster_tsp_score: {cluster_tsp_score}")
        lm.main_info(f"color_tsp_score: {color_tsp_score}")

    # Reorder cluster distance matrix to match with cluster tsp loop
    cluster_distance_tsp = np.transpose(cluster_distance.to_numpy()[cluster_tsp_path])[
        cluster_tsp_path
    ]

    # rotate color tsp loop to find the best match with cluster tsp loop
    lm.main_info(f"Optimizing cluster color mapping...")
    color_tsp_path_rotator = np.concatenate((color_tsp_path, color_tsp_path), axis=0)

    rotate_distance = 1e9
    rotate_offset = 0
    for i in range(len(color_distance)):
        color_rotate_index = color_tsp_path_rotator[i : i + len(color_distance)]
        color_distance_tsp_rotate = np.transpose(
            color_distance.to_numpy()[color_rotate_index]
        )[color_rotate_index]
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
    lm.main_info(f"Calculating cluster embedding...", indent_level=3)
    if transformation == "mds":
        from sklearn.manifold import MDS

        model = MDS(
            n_components=3,
            dissimilarity="precomputed",
            random_state=123,
        )
    elif transformation == "umap":
        from umap import UMAP

        model = UMAP(
            n_components=3,
            metric="precomputed",
            random_state=123,
        )
    embedding = model.fit_transform(cluster_distance)

    # Rescale embedding to CIE Lab colorspace
    lm.main_info(f"Rescaling embedding to CIE Lab colorspace...", indent_level=3)
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
    embedding[:, 1:3] -= 0.5
    embedding[:, 1:3] *= 200

    lm.main_info(f"Optimizing cluster color mapping...")
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
    cluster_label_mapping: List,
    cluster_label_reference: List,
) -> List:
    """
    Function to map clusters between different clustering results based
    on cluster overlap (IOU).

    Args:
        cluster_label_mapping (List): cluster result for cells to be mapped.
        cluster_label_reference (List): cluster result for cells to be
            mapped to.

    Returns:
        List: mapping result of `cluster_label_mapping`.
    """

    def iou(i, j):
        I = np.sum((cluster_label_mapping == i) & (cluster_label_reference == j))
        U = np.sum((cluster_label_mapping == i) | (cluster_label_reference == j))
        return I / U

    # Cells should be identical between different runs.
    assert len(cluster_label_mapping) == len(cluster_label_reference)

    ufunc_iou = np.frompyfunc(iou, 2, 1)
    cluster_label_mapping = np.array(cluster_label_mapping).astype(str)
    cluster_label_reference = np.array(cluster_label_reference).astype(str)

    # Reference label types should be more than mapping label types
    mapping_label_list = np.unique(cluster_label_mapping)
    reference_label_list = np.unique(cluster_label_reference)
    assert len(mapping_label_list) <= len(reference_label_list)

    # Grid label lists for vectorized calculation
    mapping_vector_column = mapping_label_list.reshape(
        1, len(mapping_label_list)
    ).repeat(len(reference_label_list), axis=0)
    reference_vector_index = reference_label_list.reshape(
        len(reference_label_list), 1
    ).repeat(len(mapping_label_list), axis=1)
    # Calculate IOU matrix
    iou_matrix = ufunc_iou(mapping_vector_column, reference_vector_index).astype(
        np.float64
    )

    # Greedy mapping to the largest IOU of each label
    relationship = {}
    index_not_mapped = np.ones(len(mapping_label_list)).astype(bool)
    iou_matrix_backup = iou_matrix.copy()
    while np.sum(iou_matrix) != 0:
        reference_index, mapping_index = np.unravel_index(
            iou_matrix.argmax(), iou_matrix.shape
        )
        relationship[mapping_label_list[mapping_index]] = reference_label_list[
            reference_index
        ]
        # Clear mapped labels to avoid duplicated mapping
        iou_matrix[reference_index, :] = 0
        iou_matrix[:, mapping_index] = 0
        index_not_mapped[mapping_index] = False

    # Check if every label is mapped to a reference
    duplicate_map_label = np.ones(len(reference_label_list))
    for mapping_index, is_force_map in enumerate(index_not_mapped):
        if is_force_map:
            reference_index = iou_matrix_backup[:, mapping_index].argmax()
            relationship[mapping_label_list[mapping_index]] = (
                reference_label_list[reference_index]
                + ".%d" % duplicate_map_label[reference_index]
            )
            duplicate_map_label[reference_index] += 1
            # Log: warning
            lm.main_warning(
                f"Mapping between cluster {mapping_label_list[mapping_index]} and cluster "
                + f"{reference_label_list[reference_index]} is not bijective.",
                indent_level=3,
            )
    mapped_cluster_label = np.frompyfunc(lambda x: relationship[x], 1, 1)(
        cluster_label_mapping
    )
    return mapped_cluster_label.tolist()
