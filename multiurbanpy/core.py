"""Compute features for a given region."""

import shutil
import tempfile
import warnings
from collections.abc import Callable, Iterable
from os import path

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import osmnx as ox
import pandas as pd
import rasterio as rio
import rasterstats
from osgeo import gdal
from pyregeon import CRSType, RegionMixin, RegionType
from rasterio import mask, transform
from tqdm import tqdm

from multiurbanpy import topo, utils

# to use `progress_apply`
tqdm.pandas()

__all__ = ["MultiScaleFeatureComputer"]


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


class MultiScaleFeatureComputer(RegionMixin):
    """Compute multi-scale features for a given region.

    Parameters
    ----------
    region : str, list-like, GeoSeries, GeoDataFrame, path-like or IO
        The region for which to compute features. This can be either:
        -  A string with a place name (Nominatim query) to geocode.
        -  A sequence with the west, south, east and north bounds.
        -  A geometric object, e.g., shapely geometry, or a sequence of geometric
            objects. In such a case, the value will be passed as the `data` argument of
            the GeoSeries constructor, and needs to be in the same CRS as the one used
            by the client's class (i.e., the `CRS` class attribute).
        -  A geopandas geo-series or geo-data frame.
        -  A filename or URL, a file-like object opened in binary ('rb') mode, or a Path
           object that will be passed to `geopandas.read_file`.
    working_dir : path-like, optional
        The directory where to store the intermediate files associated with this
        instance. If None, a temporary directory will be created.
    crs : crs-like, optional
        Coordinate reference system (CRS) to use for computing features. If None, the
        CRS will be inferred from the `region` argument.
    """

    def __init__(
        self,
        region: RegionType,
        *,
        working_dir: utils.PathType | None = None,
        building_gdf: gpd.GeoDataFrame | utils.PathType | None = None,
        crs: CRSType | None = None,
    ) -> None:
        """Initialize the feature computer object."""
        # set working directory
        if working_dir is None:
            working_dir = tempfile.mkdtemp()
            self._tmp_dir = True
        self.working_dir = working_dir

        # process crs attribute
        if crs is not None:
            self.CRS = crs

        # set region attribute
        self.region = region

        # set crs from region (if not set yet)
        if getattr(self, "CRS", None) is None:
            self.CRS = self.region.crs

    def __del__(self) -> None:
        """Destructor to clean up temporary files."""
        if getattr(self, "_tmp_dir", False):
            shutil.rmtree(self.working_dir)

    def _process_raster_filepath(self, raster_filepath, *, dst_filename="raster.tif"):
        # check that the file is in the same CRS as the region, otherwise reproject it
        # and return the path to the reprojected raster
        with rio.open(raster_filepath) as src:
            if src.crs != self.CRS:
                _raster_filepath = path.join(self.working_dir, dst_filename)
                gdal.Warp(_raster_filepath, raster_filepath, dstSRS=self.CRS)
            else:
                _raster_filepath = raster_filepath
        return _raster_filepath

    @property
    def building_gdf(self) -> gpd.GeoDataFrame | None:
        """Return the building geo-data frame."""
        try:
            return self._building_gdf
        except AttributeError:
            self._building_gdf = ox.geocode_to_gdf(
                self.region,
                buffer_dist=0,
                tags={"building": True},
                retain_invalid=True,
            )
            return self._building_gdf

    def compute_building_features(
        self,
        site_gser: gpd.GeoSeries,
        buffer_dists: Iterable[float],
        *,
        building_gdf: gpd.GeoDataFrame | utils.PathType,
    ) -> pd.DataFrame | pd.Series:
        """Compute building area (and volume if `building_gdf` has a "height" column).

        Parameters
        ----------
        site_gser : geopandas.GeoSeries
            Site locations (point geometries) to compute features.
        buffer_dists : iterable of numeric
            The buffer distances to compute features, in the same units as the tree
            canopy raster CRS.
            building_gdf : GeoDataFrame or path-like, optional
        The building geo-data frame to use. This can be either:
            -  A geopandas geo-data frame.
            -  A filename or URL, a file-like object opened in binary ('rb') mode, or a
               Path object that will be passed to `geopandas.read_file`.
            -  None, in which case the building will be downloaded from OSM using
               osmnx.
            If the geo-data frame has a "height" column, building volumes will be
            computed alongside the areas. Otherwise, only the areas will be computed.

        Returns
        -------
        building_features : pandas.DataFrame or pandas.Series
            The building features for each site (first-level index) and buffer distance
            (second-level index), as a data frame with total area ("building_area") and
            volume ("building_volume") columns if there is building height information,
            otherwise as a series of total areas.
        """

        def _compute_building_area(_building_gdf):
            return pd.Series(
                {
                    "building_area": _building_gdf["geometry"].area.sum(),
                }
            )

        def _compute_building_area_vol(_building_gdf):
            return pd.Series(
                {
                    "building_area": _building_gdf["geometry"].area.sum(),
                    "building_volume": (
                        _building_gdf["geometry"].area * _building_gdf["height"]
                    ).sum(),
                }
            )

        # process `building_gdf` arg
        if building_gdf is None:
            building_gdf = self.building_gdf
        elif not isinstance(building_gdf, gpd.GeoDataFrame):
            building_gdf = gpd.read_file(building_gdf)

        # TODO: define this at initialization?
        if "height" in building_gdf.columns:
            _compute_features = _compute_building_area_vol
        else:
            _compute_features = _compute_building_area

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
                        .sjoin(building_gdf)
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

        # TODO: DRY with `compute_building_features`
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
        tree_canopy_filepath: utils.PathType,
        site_gser: gpd.GeoSeries,
        buffer_dists: Iterable[float],
        tree_val: float,
    ) -> pd.Series:
        """Compute tree features.

        Parameters
        ----------
        tree_canopy_filepath : path-like, optional
            The path to the tree canopy raster file.
        site_gser : geopandas.GeoSeries
            Site locations (point geometries) to compute features.
        buffer_dists : iterable of numeric
            The buffer distances to compute features, in the same units as the tree
            canopy raster CRS.
        tree_val : numeric
            The value in the tree canopy raster that corresponds to tree canopy pixels.

        Returns
        -------
        tree_features_ser: pandas.Series
            The tree features for each site (first-level index) and buffer distance
            (second-level index).
        """
        # reproject if needed
        tree_canopy_filepath = self._process_raster_filepath(
            tree_canopy_filepath, dst_filename="tree-canopy.tif"
        )
        with rio.open(tree_canopy_filepath) as src:
            tree_features_ser = (
                MultiScaleFeatureComputer._multiscale_raster_stats_feature_ser(
                    src,
                    site_gser,
                    buffer_dists,
                    "sum",
                    rescale=True,
                    target_val=tree_val,
                )
            )

        return tree_features_ser.rename("tree_canopy")

    def compute_elevation_ser(
        self, dem_filepath: utils.PathType, site_gser: gpd.GeoSeries
    ) -> pd.Series:
        """Compute elevation.

        Parameters
        ----------
        dem_filepath : path-like, optional
            The path to the digital elevation model (DEM) raster file.
        site_gser : geopandas.GeoSeries
            Site locations (point geometries) to compute features.

        Returns
        -------
        elevation_ser: pandas.Series
            The elevation for each site (index).
        """
        # reproject if needed
        # TODO: how to avoid reprojecting twice for elevation and topo features?
        dem_filepath = self._process_raster_filepath(
            dem_filepath, dst_filename="dem.tif"
        )
        with rio.open(dem_filepath) as src:
            return pd.Series(
                src.read(1)[transform.rowcol(src.transform, site_gser.x, site_gser.y)],
                index=site_gser.index,
                name="elevation",
            )

    def compute_topo_features_df(
        self,
        dem_filepath: utils.PathType,
        site_gser: gpd.GeoSeries,
        buffer_dists: Iterable[float],
        *,
        topo_features: str | Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Compute topographic features.

        Parameters
        ----------
        dem_filepath : path-like, optional
            The path to the digital elevation model (DEM) raster file.
        site_gser : geopandas.GeoSeries
            Site locations (point geometries) to compute features.
        buffer_dists : iterable of numeric
            The buffer distances to compute features, in the same units as the tree
            canopy raster CRS.
        topo_features : str or iterable of str, optional
            The topographic features to compute, have to be among "slope", "northness",
            "tpi" and/or "fac". If None, all features are computed.

        Returns
        -------
        topo_features_df: pandas.DataFrame
            The topographic features (columns) for each site (first-level index) and
            buffer distance (second-level index).
        """
        # reproject if needed
        # TODO: how to avoid reprojecting twice for elevation and topo features?
        dem_filepath = self._process_raster_filepath(
            dem_filepath, dst_filename="dem.tif"
        )

        if topo_features is None:
            # do NOT compute flow accumulation by default
            topo_features = ["slope", "northness", "tpi"]
        elif isinstance(topo_features, str):
            topo_features = [topo_features]

        with rio.open(dem_filepath) as src:
            dem_res = src.res[0]
            dem_nodata = src.nodata
            # define it here to be able to set resolution/nodata in args/kwargs
            topo_features_dict = {}
            if "slope" in topo_features:
                topo_features_dict["slope"] = (
                    topo.compute_terrain_attribute,
                    ["slope_riserun", dem_res, dem_nodata, np.mean],
                    {},
                )
            if "northness" in topo_features:
                topo_features_dict["northness"] = (
                    topo.compute_terrain_attribute,
                    ["aspect", dem_res, dem_nodata, topo.northness],
                    {},
                )
            if "tpi" in topo_features:
                topo_features_dict["tpi"] = (
                    topo.comparative_height_at_center,
                    [np.mean],
                    {"nodata": dem_nodata},
                )
            if "fac" in topo_features:
                topo_features_dict["fac"] = (
                    topo.flow_accumulation_at_center,
                    [dem_res, dem_nodata],
                    {"fac_method": "D8"},
                )

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
                            func(
                                buffer_dem_arr,
                                *args,
                                **kwargs,
                            )
                            for func, args, kwargs in topo_features_dict.values()
                        ]
                    )

                return pd.DataFrame(
                    site_features,
                    index=pd.Series(buffer_dists, name="buffer_dist"),
                    columns=topo_features_dict.keys(),
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
