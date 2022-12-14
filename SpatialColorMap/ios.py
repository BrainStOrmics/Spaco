import scanpy as sc
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .color import spatial_color, spatial_color_cal
from .mat import find_neighbors

def figs(adata, palet, prefix, size):
  '''draw figures

  args:
    adata: spatial data in h5ad format
    palet: color palette
    prefix: prefix for output
    size: spot size
  '''
  sc.pl.spatial(adata, color = 'annotation', palette = palet, spot_size = 1, 
                size = size, save = prefix)

def tsp_fig(score, fsave):
  '''draw the tsp score 
  
  args:
    score: scores foreach cycle
    fsave: output
  '''
  fig, axs = plt.subplots()#figsize = (12, 9))
  axs.set_xlabel('iteration')
  axs.set_ylabel('score')
  axs.spines['top'].set_visible(False)
  axs.spines['right'].set_visible(False)
  plt.plot(score[0:2*np.argmin(score)], lw = 1)
  fig.savefig(fsave)

def rdat(data, type):
  '''loading .h5ad files
  
  args:
    data: data in h5ad format
    type: assigned cell type
  '''
  adata = sc.read_h5ad(data)
  if (not "spatial" in adata.obsm or not "annotation" in adata.obs):
    print(data + "without adata.obsm['spatial'] or adata.obs['annotation'], please supplement them first")
    sys.exit(-1)
  sample = os.path.basename(data)
  sample = sample.replace('.h5ad', '')
  return adata[np.isin(adata.obs['annotation'], type, invert = True)], sample

def assign_color(atype):
  '''deal the assigned cell type
  
  args:
    atype: assigned colors for cell type, format type1,color1;type2,color2, ..., typeN,colorN
  '''
  color = []
  type = []
  if (atype != None):
    for i in (atype.split(';')):
      tmp = i.split(',')
      type.append(tmp[0])
      color.append(tmp[1])
  return type, pd.DataFrame({'type':type, 'color':color}, index = type)

def scms(data, color, dtype, dcolor, outdir, radius = 90, knn = 4, ncell = 3, method = 'manhattan', cycle = 10000, size = 30):
  '''performded for single slice

  args:
    data: spatial data in h5ad format, multiple files separated by comma,
          <color> option is necessary for multiple files,
          adata.obsm['spatial'] and adata.obs['annotation'] are necessary
    color: a color list
    dtype: assigned colors for cell type, [type1, type2, ..., typeN]
    dcolor: a dataframe which contains the assigned information for cell type, ['type':xxx, 'color':xxx]
    outdir: outdir, default "scm_output"
    radius: radius for neighbors
    knn: the k-nearest cells
    ncell: the minimum cell number in a subnet for the specific cell type 
    method: method for matrix distance
    cycle: number of cycles
    size: spot size
  '''
  adata, sample = rdat(data, dtype)
  sc.pl.spatial(adata, color = 'annotation', spot_size = 1, size = size, save = f'_{sample}_raw.pdf')
  if (color == None):
    color = adata.uns['annotation_colors']
  tsp_score, df_color = spatial_color(color, np.array(adata.obsm['spatial']), adata.obs['annotation'], 
                                      radius, knn, ncell, method, cycle)
  df_color = pd.concat([df_color, dcolor])
  df_color.to_csv(f'{outdir}/{sample}_r{radius}k{knn}c{cycle}.csv', index = False)
  adata, sample = rdat(data, None)
  tsp_fig(tsp_score, f'{outdir}/tsp_r{radius}k{knn}c{cycle}.png')
  figs(adata, df_color['color'].tolist(), f'_{sample}_r{radius}k{knn}c{cycle}.pdf', size)

def scmm(data, color, dtype, dcolor, outdir, radius = 90, knn = 4, ncell = 3, method = 'manhattan', cycle = 10000, size = 30):
  '''performded for multiple slices

  args:
    data: spatial data in h5ad format, multiple files separated by comma,
          <color> option is necessary for multiple files,
          adata.obsm['spatial'] and adata.obs['annotation'] are necessary
    color: a color list
    dtype: assigned colors for cell type, [type1, type2, ..., typeN]
    dcolor: a dataframe which contains the assigned information for cell type, ['type':xxx, 'color':xxx]
    outdir: outdir, default "scm_output"
    radius: radius for neighbors
    knn: the k-nearest cells
    ncell: the minimum cell number in a subnet for the specific cell type 
    method: method for matrix distance
    cycle: number of cycles
    size: spot size
  '''
  tmp = data.split(',')
  df_mat = pd.DataFrame()
  ar_key = []
  for i in (tmp):
    adata, sample = rdat(i, dtype)
    key, val = find_neighbors(np.array(adata.obsm['spatial']), adata.obs['annotation'], radius, knn, ncell)
    df_tmp = pd.DataFrame(val, index = key, columns = key)
    df_sps = df_tmp.stack().reset_index()
    df_sps.columns = ['a', 'b', 'c']
    df_mat = pd.concat([df_mat, df_sps])
    ar_key.extend(key)

  '''convert 3-columns to matrix'''
  mg_mat = df_mat.groupby(by = ['a', 'b']).agg(sum).unstack().fillna(0)
  mg_mat.columns = [i[1] for i in mg_mat.columns]
  mg_key = np.array(np.unique(np.array(ar_key)))

  color = color.split(',')
  if (len(color) < len(mg_key)):
    print("at least " + str(len(mg_key)) + "colors are necessary")
    sys.exit(-1)
  tsp_score, df_color = spatial_color_cal(mg_key, np.array(mg_mat), color[0:len(mg_key)], method, cycle)
  tsp_fig(tsp_score, f'{outdir}/tsp_r{radius}k{knn}c{cycle}.png')
  df_color = pd.concat([df_color, dcolor])
  
  '''draw figures for each slice'''
  for i in (tmp):
    adata, sample = rdat(i, None)
    ss_key = np.unique(adata.obs['annotation'])
    ss_df_color = df_color.loc[ss_key,:]
    ss_df_color.to_csv(f'{outdir}/{sample}_r{radius}k{knn}c{cycle}.csv', index = False)
    figs(adata, ss_df_color['color'].tolist(), f'_{sample}_r{radius}k{knn}c{cycle}.pdf', size)


