import os

from .ios import assign_color, scmm, scms


def scm(
    data,
    color=None,
    atype=None,
    outdir="scm_output",
    radius=90,
    knn=8,
    ncell=3,
    method="manhattan",
    cycle=10000,
    size=40,
):
    """process the spatial color

    args:
      data: spatial data in h5ad format, multiple files separated by comma,
            <color> option is necessary for multiple files,
            adata.obsm['spatial'] and adata.obs['annotation'] are necessary
        adata.obsm['spatial']: contain the spatial coordinates
          array([[  4942.89432067,  -9058.15023803],
                 [  4779.1611535 ,  -9803.77175728],
                 [  4031.76501885, -11687.70673968],
                 ...,
                 [  2234.96540418, -10870.25757018],
                 [  2456.06975378, -10978.26158208],
                 [  4002.22049398, -11687.35881701]])
        adata.obs['annotation']: cell annotation
          Cell_1                  IN Pvalb+
          Cell_1000                   GN DG
          ...
          Cell_9994                      EX
          Cell_9999     Smooth muscle cells
          Name: annotation, Length: xxxxx, dtype: category
          Categories (xx, object): ['Astr1', 'Astr2', 'Astr3', ..., 'OPC', 'Olig']

      color: a color list
      atype: assign colors for some cell type, format type1,color1;type2,color2, ..., typeN,colorN
      outdir: outdir, default "scm_output"
      radius: radius for neighbors
      knn: the k-nearest cells
      ncell: the minimum cell number in a subnet for the specific cell type
      method: method for matrix distance
      cycle: number of cycles
      size: spot size
    """
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    dtype, dcolor = assign_color(atype)
    if len(data.split(",")) == 1:
        scms(
            data, color, dtype, dcolor, outdir, radius, knn, ncell, method, cycle, size
        )
    else:
        scmm(
            data, color, dtype, dcolor, outdir, radius, knn, ncell, method, cycle, size
        )
