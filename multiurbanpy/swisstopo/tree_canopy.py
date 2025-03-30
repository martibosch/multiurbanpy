"""Tree canopy."""

import os
import tempfile
from os import path

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pdal
import pooch
import rasterio as rio
from osgeo import gdal
from pystac_client.item_search import DatetimeLike
from tqdm import tqdm

from multiurbanpy import utils
from multiurbanpy.swisstopo import stac

LIDAR_TREE_VALUES = [3]
DST_OPTIONS = ["TILED:YES"]


def rasterize_lidar(
    lidar_filepath: utils.PathType,
    dst_filepath: utils.PathType,
    lidar_values: list[int],
    **gdal_writer_kwargs,
) -> str:
    """Rasterize LiDAR file."""
    pipeline = (
        pdal.Reader(lidar_filepath)
        | pdal.Filter.expression(
            expression=" || ".join(
                [f"Classification == {value}" for value in lidar_values]
            )
        )
        | pdal.Writer.gdal(
            filename=dst_filepath,
            # resolution=dst_res,
            # output_type="count",
            # data_type="int32",
            # nodata=0,
            # default_srs=stac.SWISSALTI3D_CRS,
            **gdal_writer_kwargs,
        )
    )
    _ = pipeline.execute()
    return dst_filepath


def get_tree_canopy_raster(
    region_gser: gpd.GeoSeries,
    dst_filepath: utils.PathType,
    *,
    surface3d_datetime: DatetimeLike | None = None,  # "2019/2019",
    count_threshold: int = 32,
    dst_res: float = 2,
    dst_tree_val: int = 1,
    dst_nodata: int = 0,
    dst_dtype: npt.DTypeLike = "uint32",
    lidar_tree_values: list[int] | None = None,
    rasterize_lidar_kwargs: utils.KwargsType = None,
    pooch_retrieve_kwargs: utils.KwargsType = None,
    gdal_warp_kwargs: utils.KwargsType = None,
) -> str:
    """Get tree canopy raster."""
    # use the STAC API to get the tree canopy from swissSURFACE3D
    extent_geom = region_gser.to_crs(stac.CLIENT_CRS).iloc[0]

    client = stac.SwissTopoClient()
    surface3d_gdf = client.gdf_from_collection(
        stac.SWISSSURFACE3D_COLLECTION_ID,
        extent_geom=extent_geom,
        datetime=surface3d_datetime,
    )
    # filter to get zip assets (LiDAR) only
    surface3d_gdf = surface3d_gdf[surface3d_gdf["assets.href"].str.endswith(".zip")]
    # if no datetime specified, get the latest data for each tile (location)
    if surface3d_datetime is None:
        surface3d_gdf = stac.get_latest(surface3d_gdf)

    if rasterize_lidar_kwargs is None:
        _rasterize_lidar_kwargs = {}
    else:
        _rasterize_lidar_kwargs = rasterize_lidar_kwargs.copy()

    _rasterize_lidar_kwargs.update(
        resolution=dst_res,
        output_type="count",
        data_type="uint32",
        nodata=dst_nodata,
        default_srs=stac.SWISSALTI3D_CRS,
    )
    if pooch_retrieve_kwargs is None:
        pooch_retrieve_kwargs = {}

    if lidar_tree_values is None:
        lidar_tree_values = LIDAR_TREE_VALUES

    if gdal_warp_kwargs is None:
        _gdal_warp_kwargs = {}
    else:
        _gdal_warp_kwargs = gdal_warp_kwargs.copy()
    _gdal_warp_kwargs.update(creationOptions=DST_OPTIONS)

    img_filepaths = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        # TODO: do NOT cache LiDAR files (too big), instead cache the rasterized tree
        # canopy
        _pooch_retrieve_kwargs = pooch_retrieve_kwargs.copy()
        working_dir = _pooch_retrieve_kwargs.pop("path", tmp_dir)
        for url in tqdm(surface3d_gdf["assets.href"]):
            las_filepath = pooch.retrieve(
                url,
                known_hash=None,
                processor=pooch.Unzip(),
                path=tmp_dir,
                **_pooch_retrieve_kwargs,
            )[0]  # only one file (i.e., the .las) is expected
            # we need to splitext twice because of the .las.zip extension
            img_filepath = path.join(
                working_dir,
                f"{path.splitext(path.splitext(path.basename(url))[0])[0]}.tif",
            )
            # allow resuming
            if not path.exists(img_filepath):
                # use an interim filepath to save the counts, then transform to uint8
                counts_filepath = path.join(
                    tmp_dir,
                    f"{path.splitext(path.basename(img_filepath))[0]}-counts.tif",
                )
                try:
                    _ = rasterize_lidar(
                        las_filepath,
                        counts_filepath,
                        lidar_tree_values,
                        **_rasterize_lidar_kwargs,
                    )
                except RuntimeError:
                    # some tiles may intersect with the buffered region but not contain
                    # any tree. Skip them.
                    continue
                with rio.open(counts_filepath) as src:
                    meta = src.meta.copy()
                    meta.update(dtype=dst_dtype)
                    with rio.open(img_filepath, "w", **meta) as dst:
                        dst.write(
                            np.where(
                                src.read(1) > count_threshold,
                                dst_tree_val,
                                dst_nodata,
                            ),
                            1,
                        )
                # remove the interim counts file
                os.remove(counts_filepath)
            # add path to list
            img_filepaths.append(img_filepath)

        # creationOptions=dst_options
        _ = gdal.Warp(dst_filepath, img_filepaths, format="GTiff", **_gdal_warp_kwargs)

        return dst_filepath
