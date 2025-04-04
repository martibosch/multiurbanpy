"""STAC utils."""

import warnings
from collections.abc import Iterator
from copy import deepcopy

import geopandas as gpd
import pandas as pd
import pystac_client
from pystac_client.item_search import DatetimeLike, IntersectsLike
from shapely.geometry import shape

from multiurbanpy import utils

CLIENT_URL = "https://data.geo.admin.ch/api/stac/v0.9"
# CLIENT_CRS = "EPSG:4326"  # CRS used by the client
CLIENT_CRS = "OGC:CRS84"
CH_CRS = "EPSG:2056"

SWISSALTI3D_COLLECTION_ID = "ch.swisstopo.swissalti3d"
SWISSALTI3D_CRS = "EPSG:2056"
SWISSALTI3D_RES = 0.5
SWISSALTI3D_NODATA = -9999
SWISSIMAGE10_COLLECTION_ID = "ch.swisstopo.swissimage-dop10"
SWISSIMAGE10_CRS = "EPSG:2056"
SWISSIMAGE10_RES = 0.1
SWISSIMAGE10_NODATA = 0
SWISSSURFACE3D_RASTER_COLLECTION_ID = "ch.swisstopo.swisssurface3d-raster"
SWISSSURFACE3D_RASTER_CRS = "EPSG:2056"
SWISSSURFACE3D_COLLECTION_ID = "ch.swisstopo.swisssurface3d"
SWISSSURFACE3D_CRS = "EPSG:2056"

# TODO: get CRS and resolution from collection's metadata, i.e.:
# `"summaries":{"proj:epsg":[2056],"eo:gsd":[2.0,0.1]}`
COLLECTION_CRS_DICT = {
    SWISSSURFACE3D_RASTER_COLLECTION_ID: SWISSSURFACE3D_RASTER_CRS,
    SWISSSURFACE3D_COLLECTION_ID: SWISSSURFACE3D_CRS,
    SWISSALTI3D_COLLECTION_ID: SWISSALTI3D_CRS,
}


# convert a list of STAC Items into a GeoDataFrame
# see pystac-client.readthedocs.io/en/stable/tutorials/stac-metadata-viz.html#GeoPandas
def _items_to_gdf(items: Iterator) -> gpd.GeoDataFrame:
    """Convert a list of STAC Items into a geo-data frame."""
    _items = []
    for i in items:
        _i = deepcopy(i)
        _i["geometry"] = shape(_i["geometry"])
        _items.append(_i)
    gdf = gpd.GeoDataFrame(pd.json_normalize(_items))
    for field in ["properties.datetime", "properties.created", "properties.updated"]:
        if field in gdf:
            gdf[field] = pd.to_datetime(gdf[field])
    # gdf.set_index("properties.datetime", inplace=True)
    return gdf


def _postprocess_items_gdf(items_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Swisstopo specific postprocessing of items geo-data frame."""
    base_columns = items_gdf.columns[~items_gdf.columns.str.startswith("assets.")]

    # the `json_normalize` function creates meta columns for each asset so we end up
    # with many nan columns, this function cleans them up by getting the only non-nan
    # value for each asset
    def expand_row(row):
        meta_ser = row[row.index.str.contains(row["id"])]

        split_ser = meta_ser.index.str.split(".")
        df = (
            pd.DataFrame(
                {
                    "values": meta_ser.values,
                    "meta": split_ser.str[-1].values,
                    "asset": split_ser.str[:-1].str.join("."),
                }
            )
            .pivot(columns="meta", values="values", index="asset")
            .reset_index(drop=True)
            .rename(columns=lambda col: f"assets.{col}")
        )

        return pd.concat(
            [
                _df.reset_index(drop=True)
                for _df in [
                    pd.concat(
                        [row[base_columns].to_frame().T for _ in range(len(df.index))],
                        axis="rows",
                    ),
                    df,
                ]
            ],
            axis="columns",
        )

    return pd.concat(
        [expand_row(row) for _, row in items_gdf.iterrows()], ignore_index=True
    )


def get_latest(
    gdf: gpd.GeoDataFrame,
    *,
    id_col: str = "id",
    datetime_col: str = "properties.datetime",
) -> gpd.GeoDataFrame:
    """Get the latest item for each tile."""
    return (
        gdf.sort_values(
            datetime_col,
            ascending=False,
        )
        .groupby(gdf[id_col].str.split("_").str[-1])
        .first()
        .rename_axis(index={id_col: "tile_id"})
        .reset_index()
        .set_crs(gdf.crs)
    )


class SwissTopoClient:
    """swisstopo client."""

    def __init__(self):
        """Initialize a swisstopo client."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            client = pystac_client.Client.open(CLIENT_URL)
        client.add_conforms_to("ITEM_SEARCH")
        client.add_conforms_to("COLLECTIONS")
        self._client = client

    def gdf_from_collection(
        self,
        collection_id: str,
        *,
        extent_geom: IntersectsLike | None = None,
        datetime: DatetimeLike | None = None,
        collection_extents_crs: utils.CRSType | None = None,
    ) -> gpd.GeoDataFrame:
        """Get geo-data frame of tiles of a collection."""
        if collection_extents_crs is None:
            collection_extents_crs = self._client.get_collection(
                collection_id
            ).extra_fields["crs"][0]
        search = self._client.search(
            collections=[collection_id], intersects=extent_geom, datetime=datetime
        )
        return gpd.GeoDataFrame(
            _postprocess_items_gdf(_items_to_gdf(search.items_as_dicts()))
        ).set_crs(collection_extents_crs)
