## Spaco: a comprehensive tool for coloring spatial data at single-cell resolution

[![python~=3.8](https://img.shields.io/badge/python-3.8-brightgreen)](https://www.python.org/)
[![License: GPL3.0](https://img.shields.io/badge/License-GPL3.0-yellow)](https://opensource.org/license/gpl-3-0/)
[![DOI](https://img.shields.io/badge/DOI-10.1016/j.patter.2023.100915-blue)](https://doi.org/10.1016/j.patter.2023.100915)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10113347.svg)](https://zenodo.org/doi/10.5281/zenodo.10113347)

[Quick Example](https://github.com/BrainStOrmics/Spaco_scripts/blob/main/Vignette/Demo.ipynb) - [Citation](https://www.cell.com/patterns/fulltext/S2666-3899(23)00324-0)

Visualizing spatially resolved biological data with appropriate color mapping can significantly facilitate the exploration of underlying patterns and heterogeneity. Spaco (**spa**tial **co**lorization) provides a spatially constrained approach that generates discriminate color assignments for visualizing single-cell spatial data in various scenarios.

![image](https://github.com/BrainStOrmics/Spaco/assets/37856906/cf48b6fd-6a5b-4d0d-bc90-40877ab2ff3f)

## Features

**Color assignment**

By quantifying the complex topology between cell type clusters, We optimized color assignment of to achieve better visual recognizability.

**Palette extraction**

We provide a method for extracting color plates from images. While maintaining the theme color, the color differentiation is maximized.

## Installation

```
# Latest source from github (Recommended)
pip install git+https://github.com/BrainStOrmics/Spaco.git
```

```
# PyPI
pip install spaco-release
```

## Enviroments

- python>=3.8.0
- numpy>=1.18.0
- pandas>=0.25.1
- scipy>=1.10.0
- anndata>=0.8.0
- scikit-learn>=0.19.0
- scikit-image>=0.19.0
- colormath>=3.0.0
- pyciede2000==0.0.21
- umap-learn>=0.5.0
- logging-release>=0.0.4
- typing_extensions>=4.0.0

**NOTE THAT**: Currently we found numpy version (1.22.x or 1.23.x) could influence the result of graph-guided mode of Spaco; However, colorization should be acceptable with either version; To exactly reproduce the results in Spaco [vignette](https://github.com/BrainStOrmics/Spaco_scripts/blob/main/Vignette/Demo.ipynb) or paper, please check the numpy version at the end of each jupyter notebook.


## Usage

### Quick start

```python
import spaco
import scanpy as sc # For visualization
import squidpy as sq # For loading example dataset

# loading data
adata_cell = sq.datasets.seqfish()
palette_default = adata_cell.uns['celltype_mapped_refined_colors'].copy()

# color assignment with default palette
color_mapping = spaco.colorize(
    cell_coordinates=adata_cell.obsm['spatial'],
    cell_labels=adata_cell.obs['celltype_mapped_refined'],
    palette=palette_default,
    radius=0.05,
    n_neighbors=30,
)

# Order colors by categories in adata
color_mapping = {k: color_mapping[k] for k in adata_cell.obs['celltype_mapped_refined'].cat.categories}
palette_spaco = list(color_mapping.values())

# Spaco colorization
sc.pl.spatial(adata_cell, color="celltype_mapped_refined", spot_size=0.035, palette=palette_spaco)
```

### Tutorials and demo-cases
- A brief [**demo**](https://github.com/BrainStOrmics/Spaco_scripts/blob/main/Vignette/Demo.ipynb) is included in Spaco package.
- Working with R? See [**SpacoR**](https://github.com/BrainStOrmics/SpacoR).

## Reproducibility
Scripts to reproduce benchmarking and analytic results in Spaco paper are in repository [Spaco_scripts](https://github.com/BrainStOrmics/Spaco_scripts)

## Discussion 
Users can use issue tracker to report software/code related [issues](https://github.com/BrainStOrmics/Spaco/issues). For discussion of novel usage cases and user tips, contribution on Spaco performance optimization, please contact the authors via [email](mailto:baiyinqi@genomics.cn). 
