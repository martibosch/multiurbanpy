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
from meteora.utils import CRSType, RegionType
from osgeo import gdal
from rasterio import mask, transform
from tqdm import tqdm

from multiurbanpy import topo, utils
from multiurbanpy.swisstopo import buildings, dem, tree_canopy

# to use `progress_apply`
tqdm.pandas()

__all__ = ["generate_regular_grid_gser", "MultiScaleFeatureComputer"]


# compute/geo utils

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


def generate_regular_grid_gser(
    region_gser: gpd.GeoSeries, grid_res: float, *, crs: CRSType | None = None
) -> gpd.GeoSeries:
    """
    Get a regular grid of points within a region.

    Parameters
    ----------
    region : region-like
        The region for which to generate the grid.
    grid_res : float
        The grid resolution in units of the region's CRS.
    crs : CRS-like, optional
        The CRS of the grid, required if the region is a naive geometry (without a CRS
        set), ignored otherwise.

    Returns
    -------
    grid_gser: gpd.GeoSeries
        A geo-series with the grid points.
    """
    crs = getattr(region_gser, "crs", crs)
    if crs is None:
        raise ValueError("If providing a naive geometry, the CRS must be provided.")

    def _grid_from_geom(region_geom):
        left = region_geom.bounds[0]

        top = region_geom.bounds[3]
        num_cols = int(np.ceil((region_geom.bounds[2] - left) / grid_res))
        num_rows = int(np.ceil((region_geom.bounds[1]) / grid_res))

        # generate a grid of size using numpy meshgrid
        grid_x, grid_y = np.meshgrid(
            np.arange(num_cols) * grid_res + left,
            top - np.arange(num_rows) * grid_res,
            indexing="xy",
        )

        # vectorize the grid as a geo series
        grid_gser = gpd.GeoSeries(
            gpd.points_from_xy(grid_x.flatten(), grid_y.flatten(), crs=crs)
        )

        # filter out points that are outside the agglomeration extent
        return grid_gser[grid_gser.within(region_geom)]

    return pd.concat(
        [_grid_from_geom(region_geom) for region_geom in region_gser], ignore_index=True
    ).rename_axis("grid_cell_id")


class MultiScaleFeatureComputer(RegionMixin):
    """Compute multi-scale features for a given region."""

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

    def compute_bldg_features(
        self, site_gser: gpd.GeoSeries, buffer_dists: Iterable[float]
    ) -> pd.DataFrame | pd.Series:
        """Compute building area (and volume if `bldg_gdf` has a "height" column)."""

        def _compute_bldg_area(_bldg_gdf):
            return pd.Series(
                {
                    "bldg_area": _bldg_gdf["geometry"].area.sum(),
                }
            )

        def _compute_bldg_area_vol(_bldg_gdf):
            return pd.Series(
                {
                    "bldg_area": _bldg_gdf["geometry"].area.sum(),
                    "bldg_volume": (
                        _bldg_gdf["geometry"].area * _bldg_gdf["height"]
                    ).sum(),
                }
            )

        # TODO: define this at initialization?
        if "height" in self.bldg_gdf.columns:
            _compute_features = _compute_bldg_area_vol
        else:
            _compute_features = _compute_bldg_area

        # TODO: DRY with `_multiscale_raster_feature_df`
        site_index_name = site_gser.index.name
        if site_index_name is None:
            site_index_name = "site_id"
            site_gser = site_gser.rename_axis(site_index_name)

        return (
            pd.concat(
                [
                    (
                        site_gser.buffer(buffer_dist)
                        .to_frame(name="geometry")
                        .sjoin(self.bldg_gdf)
                        .reset_index(site_index_name)
                        .groupby(by=site_index_name)
                        .progress_apply(_compute_features, include_groups=False)
                        / (np.pi * buffer_dist**2)
                    ).assign(buffer_dist=buffer_dist)
                    for buffer_dist in buffer_dists
                ],
                axis="rows",
            )
            .fillna(0)
            .set_index("buffer_dist", append=True)
            .sort_index()
        )

    @staticmethod
    def _multiscale_raster_stats_feature_ser(
        src: rio.DatasetReader,
        site_gser: gpd.GeoSeries,
        buffer_dists: Iterable[float],
        stat: str,
        *,
        rescale: bool = False,
        **arr_to_features_kwargs,
    ) -> pd.Series:
        """Compute statistics of raster values at multiple buffer distances."""
        # TODO: add support for multiple stats
        # TODO: support progress bar once related issue is solved:
        # https://github.com/perrygeo/python-rasterstats/issues/303
        features_ser = (
            pd.concat(
                [
                    pd.DataFrame(
                        rasterstats.zonal_stats(
                            site_gser.buffer(buffer_dist),
                            src.read(1),  # assume single band
                            nodata=src.nodata,
                            affine=src.transform,
                            stats=stat,
                        ),
                        index=site_gser.index,
                    ).assign(buffer_dist=buffer_dist)
                    for buffer_dist in buffer_dists
                ],
                axis="rows",
            )
            .set_index("buffer_dist", append=True)
            .sort_index()
        ).fillna(0)

        if rescale:
            # divide by buffer zone area
            features_ser = features_ser.groupby("buffer_dist")["sum"].transform(
                lambda x: x / (np.pi * float(x.name) ** 2)
            )

        return features_ser

    @staticmethod
    def _multiscale_raster_feature_df(
        src: rio.DatasetReader,
        site_gser: gpd.GeoSeries,
        buffer_dists: Iterable[float],
        arr_to_features: Callable,
        *,
        rescale: bool = False,
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

        def _compute_site_features(site_geom, **arr_to_features_kwargs):
            arr, _ = mask.mask(src, [site_geom], crop=True)
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

        # TODO: DRY with `compute_bldg_features`
        site_index_name = site_gser.index.name
        if site_index_name is None:
            site_index_name = "site_id"
            site_gser = site_gser.rename_axis(site_index_name)

        features_df = (
            pd.concat(
                [
                    _compute_site_features(site_geom, **arr_to_features_kwargs).assign(
                        **{site_index_name: site_id}
                    )
                    for site_id, site_geom in tqdm(
                        site_gser.buffer(buffer_dists[-1]).items(),
                        total=len(site_gser),
                    )
                ]
            )
            .reset_index()
            .set_index([site_index_name, "buffer_dist"])
            .sort_index()
        )

        if rescale:
            # TODO: improve column -> buffer_dist mapping
            # for column, buffer_dist in zip(columns, buffer_dists):
            #     features_df[column] /= buffer_mask_dict[buffer_dist].sum()
            features_df = features_df.apply(
                lambda column_ser: column_ser.groupby("buffer_dist").transform(
                    lambda x: x / (np.pi * float(x.name) ** 2)
                )
            )

        return features_df

    def compute_tree_features(
        self,
        site_gser: gpd.GeoSeries,
        buffer_dists: Iterable[float],
    ) -> pd.Series:
        """Compute tree features."""
        with rio.open(self.tree_canopy_filepath) as src:
            tree_features_ser = (
                MultiScaleFeatureComputer._multiscale_raster_stats_feature_ser(
                    src,
                    site_gser,
                    buffer_dists,
                    "sum",
                    rescale=True,
                    target_val=self.tree_val,
                )
            )

        return tree_features_ser.rename("tree_canopy")

    def compute_elevation_ser(self, site_gser: gpd.GeoSeries) -> pd.Series:
        """Compute elevation."""
        with rio.open(self.dem_filepath) as src:
            return pd.Series(
                src.read(1)[transform.rowcol(src.transform, site_gser.x, site_gser.y)],
                index=site_gser.index,
                name="elevation",
            )

    def compute_topo_features_df(
        self, site_gser: gpd.GeoSeries, buffer_dists: Iterable[float]
    ) -> pd.DataFrame:
        """Compute topographic features."""
        with rio.open(self.dem_filepath) as src:
            dem_res = src.res[0]
            dem_nodata = src.nodata

            @topo.no_outputs
            def dem_arr_to_topo_features(dem_arr, buffer_mask_dict):
                site_features = []

                for buffer_dist in buffer_dists:
                    try:
                        buffer_dem_arr = np.where(
                            buffer_mask_dict[buffer_dist], dem_arr, dem_nodata
                        )
                    except ValueError:
                        buffer_dem_arr = np.full(
                            buffer_mask_dict[buffer_dist].shape, dem_nodata
                        )
                        buffer_dem_arr[: dem_arr.shape[0], : dem_arr.shape[1]] = dem_arr
                    site_features.append(
                        [
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
                            topo.comparative_height_at_center(
                                np.where(
                                    buffer_dem_arr != dem_nodata, buffer_dem_arr, np.nan
                                ),
                                np.nanmean,
                            ),
                            # topo.flow_accumulation_at_center(
                            #     buffer_dem_arr, dem_res, dem_nodata
                            # ),
                        ]
                    )

                return pd.DataFrame(
                    site_features,
                    index=pd.Series(buffer_dists, name="buffer_dist"),
                    # columns=["slope", "northness", "tpi", "fac"],
                    columns=["slope", "northness", "tpi"],
                )

            with warnings.catch_warnings(), np.errstate(divide="ignore"):
                warnings.simplefilter("ignore", category=RuntimeWarning)
                return MultiScaleFeatureComputer._multiscale_raster_feature_df(
                    src,
                    site_gser,
                    buffer_dists,
                    dem_arr_to_topo_features,
                    rescale=False,
                )
