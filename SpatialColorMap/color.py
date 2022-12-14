import numpy as np
import pandas as pd
from .mat import find_neighbors, color_difference_matrix
from .dist import matrix_distance
from .tsp import tsp

def spatial_color(color, axis, type, radius = 90, knn = 4, ncell = 3, method = 'manhattan', cycle = 10000):
  '''generate the spatial color map
  
  args:
    color: an array for color
    axis: spatial axis for each cell
    type: cell type for each cell
    radius: radius for neighbors
    knn: the k-nearest cells
    ncell: the minimum cell number in a subnet for the specific cell type 
    method: method for matrix distance
    cycle: number of cycles
  '''
  mat_type, mat_adj = find_neighbors(axis, type, radius, knn, ncell)
  return spatial_color_cal(mat_type, mat_adj, color, method, cycle)

def spatial_color_cal(mat_type, mat_adj, color, method, cycle):
  ins_color = np.array(color[0:len(mat_type)])
  ins_dist = 1e9
  mat_color = color_difference_matrix(ins_color) + 1e-5
  
  mat_adj_tsp, mat_adj_tsp_score = tsp(mat_adj, cycle)
  mat_color_tsp, mat_color_tsp_score = tsp(mat_color, cycle)
  df_color = pd.DataFrame({'adj': mat_adj_tsp,  'color': mat_color_tsp}).sort_values(by = 'adj')['color']
  ins_color =  ins_color[df_color]

  arr_var = []
  
  my_idx = np.concatenate((np.arange(0, len(mat_color)), np.arange(0, len(mat_color))), axis = 0)
  for i in range(0, len(mat_color)):
    idx = my_idx[i:i+len(mat_color)]
    mat_color_sfl = np.transpose(mat_color[idx])[idx]
    dist = matrix_distance(mat_adj, mat_color_sfl, method)
    if (dist <= ins_dist):
      ins_dist = dist
      ins_color = ins_color[idx]

  return mat_adj_tsp_score, pd.DataFrame({'type': mat_type, 'color': ins_color}, index = mat_type)