"""Topographic utils."""

from collections.abc import Callable
from functools import wraps

import numpy as np
import numpy.typing as npt
import richdem as rd
from wurlitzer import pipes


def no_outputs(func):
    """Capture stdout/sderr pipes using wurlitzer."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with pipes() as (stdout, stderr):
            result = func(*args, **kwargs)
        return result

    return wrapper


class _MetaObj:
    """Meta object to set the geotransform of a richdem array."""

    def __init__(self, geotransform):
        """Initialize the meta object."""
        self.geotransform = geotransform


def compute_terrain_attribute(
    dem_arr: npt.ArrayLike,
    attrib: str,
    dst_res: float,
    dst_fill: float,
    reduce_func: Callable,
) -> float:
    """Compute a terrain attribute from a DEM array."""
    attrib_arr = rd.TerrainAttribute(
        rd.rdarray(
            dem_arr,
            meta_obj=_MetaObj([0, dst_res, 0, 0, 0, -dst_res]),
            no_data=dst_fill,
        ),
        attrib=attrib,
    )
    return reduce_func(attrib_arr[attrib_arr != dst_fill])


def northness(aspect_arr: npt.ArrayLike) -> float:
    """Compute the northness of an aspect array."""
    # with np.errstate(divide="ignore"):
    return np.mean(np.cos(np.radians(aspect_arr)))


def comparative_height_at_center(
    dem_arr: npt.ArrayLike, baseline_func: Callable
) -> float:
    """Compute a comparative height index from a DEM array around a station.

    Difference between the station elevation (assumed to be center of the buffer array)
    and the elevation within the buffer array. The baseline for comparison is computed
    with `baseline_func` function, e.g., `np.min` for the "relative height" index or
    `np.mean` for the "topographic position" index.
    """
    return dem_arr[dem_arr.shape[0] // 2, dem_arr.shape[1] // 2] - baseline_func(
        dem_arr
    )


def flow_accumulation_at_center(
    dem_arr: npt.ArrayLike, dst_res: float, dst_fill: float, fac_method: str = "D8"
) -> float:
    """Compute flow accumulation from a DEM array.

    The value at the station location (assumed to be center of the buffer array) is
    returned.
    """
    return rd.FlowAccumulation(
        rd.rdarray(
            dem_arr,
            meta_obj=_MetaObj([0, dst_res, 0, 0, 0, -dst_res]),
            no_data=dst_fill,
        ),
        method=fac_method,
    )[dem_arr.shape[0] // 2, dem_arr.shape[1] // 2]
