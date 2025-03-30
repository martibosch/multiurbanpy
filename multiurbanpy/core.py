"""Compute features for a given region."""

import hashlib
import shutil
import tempfile
import warnings
from collections.abc import Callable, Iterable
from os import path

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio as rio
import rasterstats
from meteora.mixins import RegionMixin
from meteora.utils import RegionType
from osgeo import gdal
from rasterio import mask
from tqdm import tqdm

from multiurbanpy import topo, utils
from multiurbanpy.swisstopo import buildings, dem, tree_canopy

tqdm.pandas()


# geo utils

KERNEL_DTYPE = "uint8"


def get_kernel_arr(
    kernel_radius: int, *, kernel_dtype: npt.DTypeLike = None
) -> np.ndarray:
    """Get a circular kernel."""
    # TODO: support other kernel shapes
    if kernel_dtype is None:
        kernel_dtype = KERNEL_DTYPE
    kernel_len = 2 * kernel_radius  # + 1

    y, x = np.ogrid[
        -kernel_radius : kernel_len - kernel_radius,
        -kernel_radius : kernel_len - kernel_radius,
    ]
    _mask = x * x + y * y <= kernel_radius * kernel_radius

    kernel = np.zeros((kernel_len, kernel_len), dtype=kernel_dtype)
    kernel[_mask] = 1
    return kernel


def get_buffer_mask(
    buffer_dist: float,
    src_shape: tuple[int, int],
    dst_res: float,
    *,
    kernel_dtype: npt.DTypeLike = None,
) -> np.ndarray:
    """Get a buffer mask for a given buffer distance and source data array."""
    # get kernel
    kernel_radius = int(buffer_dist / dst_res)  # in pixels
    kernel = get_kernel_arr(kernel_radius, kernel_dtype=kernel_dtype)
    height, width = src_shape[0], src_shape[1]
    kernel_height, kernel_width = kernel.shape
    # pad to match the source array shape with the kernel in the center
    return np.pad(
        kernel,
        (
            (
                (height - kernel_height) // 2,
                (height - kernel_height) // 2 + (height - kernel_height) % 2,
            ),
            (
                (width - kernel_width) // 2,
                (width - kernel_width) // 2 + (width - kernel_width) % 2,
            ),
        ),
        mode="constant",
        constant_values=0,
    )


class FeatureComputer(RegionMixin):
    """Compute features for a given region."""

    def __init__(
        self,
        region: RegionType,
        *,
        working_dir: utils.PathType | None = None,
        bldg_gdf: gpd.GeoDataFrame | utils.PathType | None = None,
        tree_canopy_filepath: utils.PathType | None = None,
        tree_val: int = 1,
        dem_filepath: utils.PathType | None = None,
        crs: utils.CRSType | None = None,
    ) -> None:
        """Initialize the feature computer object.

        Parameters
        ----------
        region : str, list-like, GeoSeries, GeoDataFrame, path-like or IO
            The region for which to compute features. This can be either:
            -  A string with a place name (Nominatim query) to geocode.
            -  A sequence with the west, south, east and north bounds.
            -  A geometric object, e.g., shapely geometry, or a sequence of geometric
                objects. In such a case, the value will be passed as the `data` argument
                of the GeoSeries constructor, and needs to be in the same CRS as the one
                used by the client's class (i.e., the `CRS` class attribute).
            -  A geopandas geo-series or geo-data frame.
            -  A filename or URL, a file-like object opened in binary ('rb') mode, or a
                Path object that will be passed to `geopandas.read_file`.
        working_dir : path-like, optional
            The directory where to store the intermediate files associated with this
            instance. If None, a temporary directory will be created.
        bldg_gdf : GeoDataFrame or path-like, optional
            The buildings geo-data frame to use. This can be either:
            -  A geopandas geo-data frame.
            -  A filename or URL, a file-like object opened in binary ('rb') mode, or a
               Path object that will be passed to `geopandas.read_file`.
            -  None, in which case the buildings will be downloaded from OSM using
               osmnx.
            If the geo-data frame has a "height" column, building volumes will be
            computed alongside the areas. Otherwise, only the areas will be computed.
        tree_canopy_filepath : path-like, optional
            The path to the tree canopy raster file. If None, it will not be possible to
            compute tree canopy features.
        tree_val : integer, default 1
            The value in the tree canopy raster that corresponds to tree canopy pixels.
        dem_filepath : path-like, optional
            The path to the digital elevation model (DEM) raster file. If None, it will
            not be possible to compute topographic features.
        crs : crs-like, optional
            Coordinate reference system (CRS) to use for computing features. If None,
            the CRS will be inferred from the `region` argument.

        """
        # set working directory
        if working_dir is None:
            working_dir = tempfile.mkdtemp()
            self._tmp_dir = True
        self.working_dir = working_dir

        # TODO: this is not necessary if we already pass the buildings, tree canopy and
        # dem
        pooch_retrieve_kwargs = {"path": working_dir}

        # process crs attribute
        if crs is not None:
            self.CRS = crs

        # set region attribute
        self.region = region

        # set crs from region (if not set yet)
        if getattr(self, "CRS", None) is None:
            self.CRS = self.region.crs

        # set a hash for the region (to identify cached results)
        self.region_hash = hashlib.sha256(self.region.to_json().encode()).hexdigest()

        # TODO: note that we should make sure that our rasters cover the entire
        # `self.region` plus the largest buffer distance

        # TODO: should building features be on-memory or on-disk?
        if bldg_gdf is not None:
            if not isinstance(bldg_gdf, gpd.GeoDataFrame):
                bldg_gdf = gpd.read_file(bldg_gdf)
            self.bldg_gdf = bldg_gdf.to_crs(self.CRS)
        else:
            bldg_gdf_filepath = path.join(
                self.working_dir, f"{self.region_hash}-buildings.gpkg"
            )
            # allow resuming
            if path.exists(bldg_gdf_filepath):
                self.bldg_gdf = gpd.read_file(bldg_gdf_filepath)
                utils.log(
                    f"Found existing buildings geo-data frame at {bldg_gdf_filepath}"
                )
            else:
                utils.log(
                    "Getting building footprints and heights from OSM and swisstopo"
                )
                self.bldg_gdf = buildings.get_bldg_gdf(
                    self.region["geometry"], **pooch_retrieve_kwargs
                )
                self.bldg_gdf.to_file(bldg_gdf_filepath)
                utils.log(f"Saved buildings geo-data frame to {bldg_gdf_filepath}")

        def _process_raster_filepath(raster_filepath, dst_filename):
            # test that the file is in the same CRS as the region
            with rio.open(raster_filepath) as src:
                if src.crs != self.CRS:
                    _raster_filepath = path.join(self.working_dir, dst_filename)
                    gdal.Warp(_raster_filepath, raster_filepath, dstSRS=self.CRS)
                else:
                    _raster_filepath = raster_filepath
            return _raster_filepath

        if tree_canopy_filepath is not None:
            self.tree_canopy_filepath = _process_raster_filepath(
                tree_canopy_filepath, f"{self.region_hash}-tree-canopy.tif"
            )
        else:
            tree_canopy_filepath = path.join(
                self.working_dir, f"{self.region_hash}-tree-canopy.tif"
            )
            # allow resuming
            if path.exists(tree_canopy_filepath):
                utils.log(
                    f"Found existing tree canopy raster at {tree_canopy_filepath}"
                )
                self.tree_canopy_filepath = tree_canopy_filepath
            else:
                utils.log("Getting tree canopy raster from swisstopo")
                self.tree_canopy_filepath = tree_canopy.get_tree_canopy_raster(
                    self.region["geometry"],
                    tree_canopy_filepath,
                    pooch_retrieve_kwargs=pooch_retrieve_kwargs,
                )
                utils.log(f"Saved tree canopy raster to {tree_canopy_filepath}")
        self.tree_val = tree_val

        if dem_filepath is not None:
            self.dem_filepath = _process_raster_filepath(
                dem_filepath, f"{self.region_hash}-dem.tif"
            )
        else:
            dem_filepath = path.join(self.working_dir, f"{self.region_hash}-dem.tif")
            # allow resuming
            if path.exists(dem_filepath):
                self.dem_filepath = dem_filepath
                utils.log(f"Found existing DEM raster at {dem_filepath}")
            else:
                utils.log("Getting DEM raster from swisstopo")
                self.dem_filepath = dem.get_dem_raster(
                    self.region["geometry"],
                    dem_filepath,
                    pooch_retrieve_kwargs=pooch_retrieve_kwargs,
                )
                utils.log(f"Saved DEM raster to {dem_filepath}")

    def __del__(self) -> None:
        """Destructor to clean up temporary files."""
        if getattr(self, "_tmp_dir", False):
            shutil.rmtree(self.working_dir)

    def compute_bldg_features_df(
        self, sample_gser: gpd.GeoSeries, buffer_dists: Iterable[float]
    ) -> pd.DataFrame:
        """Compute building area and volume."""

        def _compute_bldg_area(_bldg_gdf, buffer_dist):
            return pd.Series(
                {
                    f"bldg_area_{buffer_dist}": _bldg_gdf["geometry"].area.sum(),
                }
            )

        def _compute_bldg_area_vol(_bldg_gdf, buffer_dist):
            return pd.Series(
                {
                    f"bldg_area_{buffer_dist}": _bldg_gdf["geometry"].area.sum(),
                    f"bldg_volume_{buffer_dist}": (
                        _bldg_gdf["geometry"].area * _bldg_gdf["height"]
                    ).sum(),
                }
            )

        # TODO: define this at initialization?
        if "height" in self.bldg_gdf.columns:
            _compute_features = _compute_bldg_area_vol
        else:
            _compute_features = _compute_bldg_area
        sample_index_name = sample_gser.index.name

        return pd.concat(
            [
                sample_gser.buffer(buffer_dist)
                .to_frame(name="geometry")
                .sjoin(self.bldg_gdf)
                .reset_index(sample_index_name)
                .groupby(by=sample_index_name)
                .apply(_compute_features, buffer_dist, include_groups=False)
                / (np.pi * buffer_dist**2)
                for buffer_dist in buffer_dists
            ],
            axis="columns",
        ).fillna(0)

    @staticmethod
    def _multiscale_raster_stats_feature_df(
        src: rio.DatasetReader,
        sample_gser: gpd.GeoSeries,
        buffer_dists: Iterable[float],
        stat: str,
        *,
        rescale: bool = False,
        columns: Iterable | None = None,
        **arr_to_features_kwargs,
    ) -> pd.DataFrame:
        """Compute statistics of raster values at multiple buffer distances."""
        # TODO: add support for multiple stats
        features_df = pd.concat(
            [
                pd.DataFrame(
                    rasterstats.zonal_stats(
                        sample_gser.buffer(buffer_dist),
                        src.read(1),  # assume single band
                        nodata=src.nodata,
                        affine=src.transform,
                        stats=stat,
                    ),
                    index=sample_gser.index,
                )[stat].rename(buffer_dist)
                for buffer_dist in buffer_dists
            ],
            axis="columns",
        )
        if columns is not None:
            features_df.columns = columns
        if rescale:
            for column, buffer_dist in zip(features_df.columns, buffer_dists):
                features_df[column] /= buffer_dist**2
        return features_df

    @staticmethod
    def _multiscale_raster_feature_df(
        src: rio.DatasetReader,
        sample_gser: gpd.GeoSeries,
        buffer_dists: Iterable[float],
        arr_to_features: Callable,
        *,
        rescale: bool = False,
        columns: Iterable | None = None,
        **arr_to_features_kwargs,
    ) -> pd.DataFrame:
        """Compute features from a raster at multiple buffer distances.

        Read the raster values at the buffer distances around the geometries and apply
        `arr_to_features` to compute the features.
        """
        res = src.res
        # prepare buffer masks for each buffer distance
        buffer_mask_shape = [int(2 * buffer_dists[-1] / res_coord) for res_coord in res]
        buffer_mask_dict = {
            # assume square pixels
            buffer_dist: get_buffer_mask(buffer_dist, buffer_mask_shape, res[0])
            for buffer_dist in buffer_dists
        }

        def _compute_sample_features(sample_geom, **arr_to_features_kwargs):
            arr, _ = mask.mask(src, [sample_geom], crop=True, pad_width=1)
            try:
                return arr_to_features(
                    arr[0], buffer_mask_dict, **arr_to_features_kwargs
                )
            except ValueError:
                return arr_to_features(
                    arr[0, : buffer_mask_shape[0], : buffer_mask_shape[1]],
                    buffer_mask_dict,
                    **arr_to_features_kwargs,
                )

        if columns is None:
            columns = buffer_dists
        features_df = pd.DataFrame(
            sample_gser.buffer(buffer_dists[-1])
            .progress_apply(_compute_sample_features, **arr_to_features_kwargs)
            .to_list(),
            index=sample_gser.index,
            columns=columns,
        )
        if rescale:
            # TODO: improve column -> buffer_dist mapping
            for column, buffer_dist in zip(columns, buffer_dists):
                features_df[column] /= buffer_mask_dict[buffer_dist].sum()

        return features_df

    def compute_tree_features_df(
        self,
        sample_gser: gpd.GeoSeries,
        buffer_dists: Iterable[float],
    ) -> pd.DataFrame:
        """Compute tree features."""
        with rio.open(self.tree_canopy_filepath) as src:
            tree_features_df = FeatureComputer._multiscale_raster_stats_feature_df(
                src,
                sample_gser,
                buffer_dists,
                "sum",
                rescale=True,
                columns=[f"tree_canopy_{buffer_dist}" for buffer_dist in buffer_dists],
                target_val=self.tree_val,
            )

        return tree_features_df

    def compute_topo_features_df(
        self, sample_gser: gpd.GeoSeries, buffer_dists: Iterable[float]
    ) -> pd.DataFrame:
        """Compute topographic features."""
        with rio.open(self.dem_filepath) as src:
            dem_res = src.res[0]
            dem_nodata = src.nodata

            @topo.no_outputs
            def dem_arr_to_topo_features(dem_arr, buffer_mask_dict):
                sample_features = []

                for buffer_dist in buffer_dists:
                    buffer_dem_arr = np.where(
                        buffer_mask_dict[buffer_dist], dem_arr, dem_nodata
                    )
                    sample_features += [
                        topo.compute_terrain_attribute(
                            buffer_dem_arr,
                            "slope_riserun",
                            dem_res,
                            dem_nodata,
                            np.mean,
                        ),
                        topo.compute_terrain_attribute(
                            buffer_dem_arr,
                            "aspect",
                            dem_res,
                            dem_nodata,
                            topo.northness,
                        ),
                        topo.comparative_height_at_center(buffer_dem_arr, np.mean),
                        topo.flow_accumulation_at_center(
                            buffer_dem_arr, dem_res, dem_nodata
                        ),
                    ]

                return sample_features

            with warnings.catch_warnings(), np.errstate(divide="ignore"):
                warnings.simplefilter("ignore", category=RuntimeWarning)
                return FeatureComputer._multiscale_raster_feature_df(
                    src,
                    sample_gser,
                    buffer_dists,
                    dem_arr_to_topo_features,
                    rescale=False,
                    # ACHTUNG: the order of the column names must match how the features
                    # are computed in `dem_arr_to_topo_features`
                    columns=[
                        f"{feature}_{buffer_dist}"
                        for buffer_dist in buffer_dists
                        for feature in [
                            "slope",
                            "northness",
                            "tpi",
                            "fac",
                        ]
                    ],
                )
