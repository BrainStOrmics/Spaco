from typing import List, Tuple, Union

import numpy as np
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, XYZColor, sRGBColor
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_local_search

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def matrix_distance(
    matrix_x: np.ndarray,
    matrix_y: np.ndarray,
    metric: Literal["euclidean", "manhattan", "log"] = "manhattan",
) -> float:
    """
    Calculate the distance between two matrices.

    Args:
        matrix_x (np.ndarray): matrix x.
        matrix_y (np.ndarray): matrix y.
        metric (Literal[&quot;euclidean&quot;, &quot;manhattan&quot;, &quot;log&quot;], optional): metric used to calculate distance. Defaults to "manhattan".

    Returns:
        float: the distance between two matrices.
    """

    assert matrix_x.shape == matrix_y.shape, "Matrices should be of same size"

    if metric.lower() == "euclidean":
        return np.linalg.norm(matrix_x - matrix_y)
    elif metric.lower() == "manhattan":
        return np.sum(abs(matrix_x - matrix_y))
    elif metric.lower() == "log":
        return np.sum(matrix_x * np.log(matrix_y))


def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    """
    Convert hex string to RGB value (0~255).

    Args:
        hex_code (str): hex string representing a RGB color.

    Returns:
        Tuple[int,int,int]: integer values for RGB channels.
    """

    hex_code = hex_code.lstrip("#")
    assert (
        len(hex_code) == 6
    ), "Currently only support vanilla RGB hex color without alpha."

    return tuple(int(hex_code[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb_code: Union[Tuple, List]) -> str:
    """
    Convert RGB value (0~255) to hex string.

    Args:
        rgb_code (Union[Tuple, List]): RGB channel value (scaled to 0~255).

    Returns:
        str: color hex string.
    """

    return sRGBColor(
        rgb_code[0], rgb_code[1], rgb_code[2], is_upscaled=True
    ).get_rgb_hex()


def lab_to_hex(lab_code: Union[Tuple, List]) -> str:
    """
    Convert CIE Lab color value to hex string.
    see `Lab color space <http://en.wikipedia.org/wiki/Lab_color_space>` on Wikipedia.

    Args:
        lab_code (Union[Tuple, List]): CIE Lab color value.

    Returns:
        str: color hex string.
    """

    # Convert Lab to sRGB
    lab_object = LabColor(lab_code[0], lab_code[1], lab_code[2], illuminant="d65")
    xyz_object = convert_color(lab_object, XYZColor)
    rgb_object = convert_color(xyz_object, sRGBColor)

    # Clip to legal RGB color
    rgb_object.rgb_r = min(rgb_object.rgb_r, 1)
    rgb_object.rgb_g = min(rgb_object.rgb_g, 1)
    rgb_object.rgb_b = min(rgb_object.rgb_b, 1)
    rgb_object.rgb_r = max(rgb_object.rgb_r, 0)
    rgb_object.rgb_g = max(rgb_object.rgb_g, 0)
    rgb_object.rgb_b = max(rgb_object.rgb_b, 0)

    return rgb_object.get_rgb_hex()


def color_difference_rgb(color_x: str, color_y: str) -> float:
    """
    Calculate the perceptual difference between colors.
    #TODO: Add a Wiki of this calculation

    Args:
        color_x (str): hex string for color x.
        color_y (str): hex string for color y.

    Returns:
        float: the perceptual difference between two colors.
    """

    rgb_x = hex_to_rgb(color_x)
    rgb_y = hex_to_rgb(color_y)

    rgb_r = (rgb_x[0] + rgb_y[0]) / 2

    return (
        (2 + rgb_r / 256) * (rgb_x[0] - rgb_y[0]) ** 2
        + 4 * (rgb_x[1] - rgb_y[1]) ** 2
        + (2 + (255 - rgb_r) / 256) * (rgb_x[2] - rgb_y[2]) ** 2
    ) ** 0.5


def tsp(
    distance_matrix: np.ndarray,
    solver_backend: Literal["exact", "heuristic"] = "heuristic",
) -> Tuple[List[int], float]:
    """
    TSP solver. <https://github.com/fillipe-gsm/python-tsp>

    Args:
        distance_matrix (np.ndarray): distance adjacent matrix.
        solver_backend (Literal[&quot;exact&quot;, &quot;heuristic&quot;], optional): exact solver guarantees
            reproducibility but takes more time and memory, heuristic solver is fast but may fall into local
            optimal and does not guarantee reproducibility. Defaults to "heuristic".

    Returns:
        Tuple[List[int], float]: tsp_path is a list of vertices, tsp_score is the length of path.
    """

    if solver_backend == "exact":
        tsp_path, tsp_score = solve_tsp_local_search(distance_matrix)
    elif solver_backend == "heuristic":
        tsp_path, tsp_score = solve_tsp_dynamic_programming(distance_matrix)

    return tsp_path, tsp_score


def extract_palette(
    reference_image: np.ndarray,
    n_colors: int,
) -> List[str]:
    """
    Extract palette from image. Implemented based on #TODO: github link if available.

    Reference:

    Args:
        reference_image (np.ndarray): _description_
        n_colors (int): _description_

    Returns:
        List[str]: _description_
    """

    assert False, "under development."  # TODO: implement here

    return None
