[![PyPI version fury.io](https://badge.fury.io/py/multiurbanpy.svg)](https://pypi.python.org/pypi/multiurbanpy/)
[![Documentation Status](https://readthedocs.org/projects/multiurbanpy/badge/?version=latest)](https://multiurbanpy.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/martibosch/multiurbanpy/main.svg)](https://results.pre-commit.ci/latest/github/martibosch/multiurbanpy/main)
[![codecov](https://codecov.io/gh/martibosch/multiurbanpy/branch/main/graph/badge.svg?token=hKoSSRn58a)](https://codecov.io/gh/martibosch/multiurbanpy)
[![GitHub license](https://img.shields.io/github/license/martibosch/multiurbanpy.svg)](https://github.com/martibosch/multiurbanpy/blob/main/LICENSE)

# multiurbanpy

Computing multi-scale urban features (building areas and volumes, tree canopy, terrain/topographic indices...) in Python.

> *multiurbanpy* is an analogy to the [multilandr](https://github.com/phuais/multilandR) package [1] to compute multi-scale landscape metrics in R. However, instead of lansdcape metrics<sup>[1](#pylandstats)</sup>, multiurbanpy computes mutli-scale metrics for urban landscapes such as building areas and volumes, tree canopy cover or topographic features.

Example application to compute the proportion of tree canopy around (with multiple buffer radii) weather stations in Zurich, Switzerland:

![stations-tree-canopy](https://github.com/martibosch/multiurbanpy/raw/main/figures/stations-tree-canopy.png)

*(C) OpenStreetMap contributors, tiles style by Humanitarian OpenStreetMap Team hosted by OpenStreetMap France*

## Overview

Start by instantiating a `MultiScaleFeatureComputer` for [a given region of interest](https://github.com/martibosch/pyregeon). Then, given a list of site locations, you can compute urban features at multiple scales, i.e., based on the landscape surrounding each site for multiple buffer radii:

```python
import swisstopopy

import multiurbanpy as mup

# parameters
region = "EPFL"
crs = "epsg:2056"
buffer_dists = [10, 25, 50, 100]
grid_res = 200

# instantiate the multi-scale feature computer
msfc = mup.MultiScaleFeatureComputer(region=region, crs=crs)

# generate a regular grid of points/sites within the region
site_gser = msfc.generate_regular_grid_gser(grid_res, geometry_type="point")

# get a tree canopy raster from swisstopo data
dst_filepath = "tree-canopy.tif"
swisstopopy.get_tree_canopy_raster(region, dst_filepath)
tree_val = 1  # pixel value representing a tree in the canopy raster

# generate a DEM raster from swisstopo data
dem_filepath = "dem.tif"
swisstopopy.get_dem_raster(region, dem_filepath)

# compute multi-scale features

# building areas from OpenStreetMap buildings (via osmnx)
features_df = pd.concat(
    [
        msfc.compute_building_features(site_gser, buffer_dists),
        msfc.compute_tree_features(
            tree_canopy_filepath, site_gser, buffer_dists, tree_val
        ),
        msfc.compute_topo_features_df(
            dem_filepath,
            site_gser,
            buffer_dists,
        ),
    ],
    axis="columns",
)
features_df.head()
```

| grid_cell_id | buffer_dist | building_area | tree_canopy |    slope | northness |       tpi |
| -----------: | ----------: | ------------: | ----------: | -------: | --------: | --------: |
|            0 |          10 |    313.654849 |    0.000000 | 0.020963 |  0.180932 | -0.006561 |
|              |          25 |   3920.685613 |    0.014260 | 0.052408 |  0.023872 |  0.036682 |
|              |          50 |  31365.484905 |    0.047746 | 0.070575 | -0.006432 | -0.075104 |
|              |         100 | 250923.879244 |    0.043386 | 0.073637 |  0.006363 | -0.716217 |
|            1 |          10 |    627.309698 |    0.000000 | 0.095521 |  0.228504 |  0.080963 |

See the [overview notebook](https://multiurbanpy.readthedocs.io/en/latest/overview.html) and the [API documentation](https://multiurbanpy.readthedocs.io/en/latest/api.html) for more details on the features of multiurbanpy.

## Installation

Like many other geospatial Python packages, multiurbanpy requires many base C libraries that cannot be installed with pip. Accordingly, the best way to install multiurbanpy is to use conda/mamba, i.e., in a given conda environment, run:

```bash
# or mamba install -c conda-forge geopandas
conda install -c conda-forge geopandas
```

Within the same conda environment, you can then install multiurbanpy using pip:

```bash
pip install https://github.com/martibosch/multiurbanpy/archive/main.zip
```

## Acknowledgements

- This package was created with the [martibosch/cookiecutter-geopy-package](https://github.com/martibosch/cookiecutter-geopy-package) project template.

## Footnotes

<a name="pylandstats">1</a>. You can use the [`MultiScaleAnalysis`](https://github.com/martibosch/pylandstats-notebooks/blob/main/notebooks/06-multiscale-analysis.ipynb) class of [pylandstats](https://github.com/martibosch/pylandstats) [2] to compute multi-scale landscape metrics in Python.

## References

1. Huais, P. Y. (2024). Multilandr: An r package for multi-scale landscape analysis. Landscape Ecology, 39(8), 140.
1. Bosch, M. (2019). PyLandStats: An open-source Pythonic library to compute landscape metrics. PloS one, 14(12), e0225734.
