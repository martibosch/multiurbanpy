"""Digital elevation model."""

import geopandas as gpd
import pooch
from osgeo import gdal
from pystac_client.item_search import DatetimeLike
from tqdm import tqdm

from multiurbanpy import utils
from multiurbanpy.swisstopo import stac

DST_OPTIONS = ["TILED:YES"]


def get_dem_raster(
    region_gser: gpd.GeoSeries,
    dst_filepath: utils.PathType,
    *,
    alti3d_datetime: DatetimeLike | None = None,
    alti3d_res: float = 2,
    pooch_retrieve_kwargs: utils.KwargsType = None,
    gdal_warp_kwargs: utils.KwargsType = None,
) -> str:
    """Get tree canopy raster."""
    # use the STAC API to get the tree canopy from swissSURFACE3D
    extent_geom = region_gser.to_crs(stac.CLIENT_CRS).iloc[0]

    client = stac.SwissTopoClient()
    alti3d_gdf = client.gdf_from_collection(
        stac.SWISSALTI3D_COLLECTION_ID,
        extent_geom=extent_geom,
        datetime=alti3d_datetime,
    )

    # filter to get tiff images only
    alti3d_gdf = alti3d_gdf[alti3d_gdf["assets.href"].str.endswith(".tif")]
    # filter to get the resolution data at the specified resolution
    alti3d_gdf = alti3d_gdf[alti3d_gdf["assets.eo:gsd"] == alti3d_res]
    # if no datetime specified, get the latest data for each tile (location)
    if alti3d_datetime is None:
        alti3d_gdf = stac.get_latest(alti3d_gdf)

    if pooch_retrieve_kwargs is None:
        pooch_retrieve_kwargs = {}

    if gdal_warp_kwargs is None:
        _gdal_warp_kwargs = {}
    else:
        _gdal_warp_kwargs = gdal_warp_kwargs.copy()
    _gdal_warp_kwargs.update(creationOptions=DST_OPTIONS)

    img_filepaths = []
    for url in tqdm(
        alti3d_gdf["assets.href"],
    ):
        img_filepath = pooch.retrieve(url, known_hash=None, **pooch_retrieve_kwargs)
        img_filepaths.append(img_filepath)
    _ = gdal.Warp(dst_filepath, img_filepaths, format="GTiff", **_gdal_warp_kwargs)
