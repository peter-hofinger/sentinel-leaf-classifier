"""Functions for processing satellite data tailored to leaf type mixture analysis.

Google Earth Engine is used to retrieve Sentinel-2 satellite images and compute composites. The composites are saved as GeoTIFFs. The composites can be used as input data for machine learning models. The labels are computed from plot data on the individual tree level and saved as GeoTIFFs. The labels can be used as target data for machine learning models. For inference you can convert a Shapefile to a raster mask, for which you can then compute a composite.

Typical usage example:

    import pandas as pd
    from ltm.data import compute_label, sentinel_composite, shapefile2raster
    from datetime import datetime

    plot = pd.read_csv("plot.csv")

    target_path = "target.tif"
    data_path = "data.tif"

    compute_label(
        target_path=target_path,
        plot=plot,
    )

    sentinel_composite(
        target_path_from=target_path,
        data_path_to=data_path,
        time_window=(datetime(2020, 1, 1), datetime(2020, 12, 31)),
    )

    shapefile2raster(
        shapefile_path="shapefile.shp",
        raster_path="mask.tif",
    )
"""

import asyncio
import datetime
from functools import lru_cache
from itertools import product
from numbers import Number
from pathlib import Path
from typing import Any, Coroutine, List, Tuple
from urllib.request import urlopen

import aiohttp
import ee
import eemont
import geopandas as gpd
import nest_asyncio
import numpy as np
import pandas as pd
import rasterio
import utm
from pyproj import CRS
from rasterio.io import MemoryFile
from tqdm.notebook import tqdm
from typeguard import typechecked

BROADLEAF_AREA = "Broadleaf Area"
CONIFER_AREA = "Conifer Area"
CONIFER_PROPORTION = "Conifer Proportion"
CONIFER = "conifer"
DBH = "dbh"
LATITUDE = "latitude"
LONGITUDE = "longitude"
SCALE = 10  # Fine Sentinel-2 resolution in meters


@typechecked
def _initialize_ee() -> None:
    try:
        getattr(ee.Reducer, "mean")
    except AttributeError:
        print("Initializing Earth Engine API...")
        ee.Initialize()


@typechecked
def _check_time_window(
    time_window: Tuple[datetime.date, datetime.date],
    level_2a: bool,
) -> None:
    start, end = [round(time.timestamp() * 1000) for time in time_window]
    if start >= end:
        raise ValueError(
            f"start ({time_window[0]}) must be before end ({time_window[1]}) of timewindow"
        )
    if level_2a and time_window[0] < datetime.datetime(2017, 3, 28):
        if time_window[0] >= datetime.datetime(2015, 6, 27):
            raise ValueError(
                "Level-2A data is not available before 2017-03-28. Use Level-1C data instead."
            )

        raise ValueError("Level-2A data is not available before 2017-03-28.")
    if not level_2a and time_window[0] < datetime.datetime(2015, 6, 27):
        raise ValueError("Level-1C data is not available before 2015-06-27.")


@typechecked
def _check_items(
    items: list | None,
    valid_items: list,
    items_desc: str,
    within_desc: str,
) -> None:
    if items is None:
        return

    if len(set(items)) == len(items):
        return

    duplicates = [item for item in set(items) if items.count(item) > 1]
    if duplicates:
        raise ValueError(f"Duplicate {items_desc}: {', '.join(duplicates)}")

    invalid_items = [item for item in items if item not in valid_items]
    if invalid_items:
        raise ValueError(
            f"Invalid {items_desc} not in {within_desc}: {', '.join(invalid_items)}"
        )


@typechecked
def _check_path(
    *paths: str,
    suffix: str,
    check_parent: bool = True,
    check_self: bool = True,
) -> None:
    for path in paths:
        pathlib = Path(path)
        if pathlib.suffix != suffix:
            raise ValueError(f"{path} must end with {suffix}")
        if check_parent and not pathlib.parent.exists():
            raise ValueError(f"{path} parent directory does not exist")
        if check_self and not pathlib.exists():
            raise ValueError(f"{path} does not exist")


@typechecked
def _check_band_limit(
    sentinel_bands: List[str] | None,
    level_2a: bool,
    indices: List[str] | None,
    temporal_reducers: List[str] | None,
    num_composites: int,
) -> None:
    # Compute number of bands
    if sentinel_bands is None:
        num_bands = len(list_bands(level_2a))
    else:
        num_bands = len(sentinel_bands)

    if indices is not None:
        num_bands += len(indices)

    # Compute number of reducers
    if temporal_reducers is None:
        num_reducers = 1
    else:
        num_reducers = len(temporal_reducers)

    total_bands = num_bands * num_reducers * num_composites
    if total_bands > 5000:
        raise ValueError(
            f"You exceed the 5000 bands max limit of GEE: {total_bands} bands"
        )


@typechecked
def _sentinel_crs(
    latitude: float,
    longitude: float,
) -> str:
    zone_number = utm.latlon_to_zone_number(latitude, longitude)
    is_south = utm.latitude_to_zone_letter(latitude) < "N"

    crs = CRS.from_dict({"proj": "utm", "zone": zone_number, "south": is_south})
    crs = f"EPSG:{crs.to_epsg()}"

    return crs


@typechecked
def _split_time_window(
    time_window: Tuple[datetime.date, datetime.date],
    num_splits: int,
    level_2a: bool,
) -> List[Tuple[datetime.date, datetime.date]]:
    _check_time_window(time_window, level_2a)

    start = time_window[0]
    end = time_window[1]
    delta = (end - start + datetime.timedelta(days=1)) / num_splits

    if delta.days < 1:
        raise ValueError(
            f"Time window {time_window} is too small to split into {num_splits} sub windows"
        )

    sub_windows = []
    for i in range(num_splits):
        sub_window = (
            start + i * delta,
            start + (i + 1) * delta - datetime.timedelta(days=1),
        )
        sub_windows.append(sub_window)

    return sub_windows


@typechecked
def _get_roi_scale_crs(
    target_path: str,
) -> Tuple[ee.Geometry, float, str]:
    _check_path(target_path, suffix=".tif")

    # Get region of interest (ROI), scale, and coordinate reference system (CRS)
    with rasterio.open(target_path) as src:
        crs = src.crs.to_string()
        res = src.res
        if res[0] != res[1]:
            raise ValueError("resolution is not square!")
        scale = res[0]
        bounds = src.bounds

    fc = ee.FeatureCollection(
        [
            ee.Geometry.Point([x, y], proj=crs)
            for x, y in product(bounds[::2], bounds[1::2])
        ]
    )
    roi = fc.geometry().bounds(0.01, proj=crs)

    return roi, scale, crs


@typechecked
def _select_bands(
    s2_window: ee.ImageCollection,
    sentinel_bands: List[str] | None,
    indices: List[str] | None,
    level_2a: bool,
    remove_clouds: bool,
) -> ee.ImageCollection:
    # Check sentinel_bands and indices
    within_desc = "Level 2A" if level_2a else "Level 1C"
    _check_items(sentinel_bands, list_bands(level_2a), "sentinel_bands", within_desc)
    _check_items(indices, list_indices(), "indices", "eemont package")

    # Combine bands and indices
    if sentinel_bands is None:
        bands = list_bands(level_2a)
    else:
        bands = sentinel_bands.copy()
    if indices is not None:
        bands += indices

    if s2_window.size().getInfo() > 0:
        # Remove clouds
        if remove_clouds:
            s2_window = s2_window.map(_mask_s2_clouds)
            if level_2a:
                s2_window = s2_window.map(_mask_level_2a)
        s2_window = s2_window.map(lambda image: image.divide(10000))

        # Add indices before possibly removing bands necessary for computing indices
        if indices:
            s2_window = s2_window.spectralIndices(indices)

        # Select bands
        s2_window = s2_window.select(bands)
    else:
        # Handle empty image collection
        masked_image = ee.Image.constant([0] * len(bands)).rename(bands)
        masked_image = masked_image.mask(masked_image)
        s2_window = ee.ImageCollection([masked_image])

    return s2_window


@typechecked
def _reduce_window(
    s2_window: ee.ImageCollection,
    temporal_reducers: List[str] | None,
) -> ee.Image:
    _check_items(temporal_reducers, list_reducers(), "temporal_reducers", "ee.Reducer")

    # Default to mean reducer
    if not temporal_reducers:
        temporal_reducers = ["mean"]

    # Reduce by temporal_reducers
    reduced_images = []
    for temporal_reducer in temporal_reducers:
        reducer = getattr(ee.Reducer, temporal_reducer)()
        # Check if reducer is bitwise, if so, convert to integer type
        if temporal_reducer.startswith("bitwise"):
            reduced_image = s2_window.map(lambda image: image.toInt()).reduce(reducer)
        else:
            reduced_image = s2_window.reduce(reducer)

        band_names = reduced_image.bandNames().getInfo()
        if list_reducer_bands(temporal_reducer) is None:
            band_names = [
                f"{_split_reducer_band_name(band_name)[0]}_{temporal_reducer}"
                for band_name in reduced_image.bandNames().getInfo()
            ]
        else:
            # Handle reducers like kendallsCorrelation with multiple outputs
            new_band_names = []
            for band_name in band_names:
                band, reducer_label = _split_reducer_band_name(band_name)
                new_band_name = f"{band}_{temporal_reducer}_{reducer_label}"
                new_band_names.append(new_band_name)
            band_names = new_band_names

        reduced_image = reduced_image.rename(band_names)

        reduced_images.append(reduced_image)

    # Combine reduced_images into one image
    datum = ee.ImageCollection(reduced_images).toBands()

    return datum


@typechecked
def _split_reducer_band_name(
    band_name: str,
) -> Tuple[str, str]:
    valid_bands = set(list_bands(level_2a=True))
    valid_bands = valid_bands.union(set(list_bands(level_2a=False)))
    valid_bands = valid_bands.union(set(list_indices()))

    parts = band_name.split("_")
    partial_band = parts[0]
    band = partial_band if partial_band in valid_bands else ""
    for part in parts[1:]:
        partial_band += f"_{part}"
        if partial_band in valid_bands:
            band = partial_band

    reducer = band_name[len(band) + 1 :]

    return band, reducer


@typechecked
def _prettify_band_names(image: ee.Image) -> ee.Image:
    band_names = image.bandNames().getInfo()

    pretty_names = []
    for band_name in band_names:
        composite_idx, _, band_reducer = band_name.split("_", 2)
        band_label, reducer_string = _split_reducer_band_name(band_reducer)

        reducer_parts = reducer_string.split("_")
        reducer_label = reducer_parts[0]
        reducer_band = None if len(reducer_parts) == 1 else reducer_parts[1]

        pretty_name = combine_band_name(
            int(composite_idx) + 1, band_label, reducer_label, reducer_band
        )

        pretty_names.append(pretty_name)

    image = image.rename(pretty_names)

    return image


@typechecked
async def _fetch(
    url: str,
) -> bytes:
    async with aiohttp.ClientSession(read_timeout=60 * 60) as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise ValueError(f"Failed to fetch {url}")
            return await response.read()


@typechecked
async def _wrap_coroutine(
    idx: int,
    coroutine: Coroutine,
) -> Tuple[int, Any]:
    result = await coroutine

    return idx, result


@typechecked
async def _async_gather(
    *coroutines: Coroutine,
    desc: str | None = None,
) -> list[Any]:
    # Without tqdm
    if len(coroutines) < 2:
        return await asyncio.gather(*coroutines)

    with tqdm(total=len(coroutines), desc=desc) as pbar:
        wrapper = [
            _wrap_coroutine(idx, coroutine) for idx, coroutine in enumerate(coroutines)
        ]
        results = [None] * len(coroutines)
        for future in asyncio.as_completed(wrapper):
            i, result = await future
            results[i] = result
            pbar.update(1)

    return results


@typechecked
def _gather(
    *coroutines: Coroutine,
    desc: str | None = None,
    asynchronous: bool = True,
) -> list[Any]:
    nest_asyncio.apply()
    if asynchronous:
        return asyncio.run(_async_gather(*coroutines, desc=desc))

    # Create iterable
    if len(coroutines) < 2:
        iterable = coroutines
    else:
        iterable = tqdm(coroutines, desc=desc)

    # Run synchronously
    results = []
    for coroutine in iterable:
        results.append(asyncio.run(coroutine))

    return results


@typechecked
async def _fetch_image(
    image: ee.Image,
) -> bytes:
    download_params = {
        "scale": image.projection().nominalScale().getInfo(),
        "crs": image.projection().getInfo()["crs"],
        "format": "GEO_TIFF",
    }
    url = image.getDownloadURL(download_params)

    return await _fetch(url)


@typechecked
def _responses2data(
    image: bytes,
    mask: bytes,
) -> Tuple[rasterio.profiles.Profile, np.ndarray]:
    with MemoryFile(image) as memfile, MemoryFile(mask) as mask_memfile:
        with memfile.open() as dataset, mask_memfile.open() as mask_dataset:
            # Get profile and set nodata to np.nan
            profile = dataset.profile
            profile["nodata"] = np.nan

            # Read and mask raster
            raster = dataset.read()
            mask_raster = mask_dataset.read()
            raster[mask_raster == 0] = np.nan

    return profile, raster


@typechecked
async def _get_image_data(
    image: ee.Image,
    bands: list[str] | None = None,
) -> Tuple[rasterio.profiles.Profile, np.ndarray] | None:
    # Select bands
    if bands is not None:
        image = image.select(bands)

    try:
        # Download image and mask
        image = image.toDouble()
        image_response = await _fetch_image(image)
        mask_response = await _fetch_image(image.mask())
    except ee.ee_exception.EEException as exc:
        raise ValueError(
            "Failed to compute image. A small batch_size might fix this error."
        ) from exc

    # Get profile and raster
    profile, raster = _responses2data(
        image_response,
        mask_response,
    )

    return profile, raster


@typechecked
def _save_image(
    image: ee.Image,
    file_path: str,
    batch_size: int | None = None,
) -> None:
    # Check file path and batch size
    _check_path(file_path, suffix=".tif", check_self=False)
    if not (batch_size is None or batch_size > 0):
        raise ValueError("batch_size must be a positive integer or None")

    # Get image data
    profile = None
    image_raster = None
    bands = image.bandNames().getInfo()

    # Create batches of bands
    if batch_size is None:
        batch_size = len(bands)
    band_batches = [bands[i : i + batch_size] for i in range(0, len(bands), batch_size)]

    # Warn user of GEE quota limits
    if len(band_batches) > 40:
        print(
            "Warning: Google Earth Engine usually has a quota limit of 40 concurrent requests. Consider increasing batch_size to reduce the number of batches."
        )

    # Get all image data using gather()
    try:
        coroutines = [_get_image_data(image, bands=batch) for batch in band_batches]
        results = _gather(*coroutines, desc="Batches")
    except (aiohttp.ClientError, ValueError) as exc:
        if len(band_batches) == 1:
            raise exc
        print("Failed to download asynchroniously. Trying synchroniously...")
        coroutines = [_get_image_data(image, bands=batch) for batch in band_batches]
        results = _gather(*coroutines, desc="Batches", asynchronous=False)

    # Combine results
    profile = results[0][0]
    image_raster = np.concatenate([raster for _, raster in results], axis=0)

    # Save image
    profile["count"] = len(bands)
    with rasterio.open(file_path, "w", **profile) as dst:
        dst.write(image_raster)
        dst.descriptions = bands

    print(f"GeoTIFF saved as {file_path}")


@typechecked
def _mask_s2_clouds(
    image: ee.Image,
) -> ee.Image:
    qa = image.select("QA60")

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0)
    mask = mask.And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))

    return image.updateMask(mask)


@typechecked
def _mask_level_2a(
    image: ee.Image,
) -> ee.Image:
    # Mask clouds
    cloud_prob = image.select("MSK_CLDPRB")
    cloud_prob = cloud_prob.unmask(sameFootprint=False)
    mask = cloud_prob.eq(0)

    # Mask snow
    snow_prob = image.select("MSK_SNWPRB")
    snow_prob = snow_prob.unmask(sameFootprint=False)
    mask = mask.And(snow_prob.eq(0))

    return image.updateMask(mask)


@typechecked
def _compute_area(
    fc: ee.FeatureCollection,
    scale: float,
    fine_scale: float,
    crs: str,
) -> ee.Image:
    # Compute area by masking a constant image and multiplying by scale**2
    fc_area = ee.Image.constant(1).clip(fc).mask()
    fc_area = ee.Image.constant(scale**2).multiply(fc_area)

    # Reduce to coarse resolution
    fc_area = fc_area.reproject(scale=fine_scale, crs=crs)
    fc_area = fc_area.reduceResolution(ee.Reducer.mean(), maxPixels=10_000)

    return fc_area


@typechecked
def _check_plot(
    plot: pd.DataFrame,
) -> pd.DataFrame:
    plot = plot.rename(columns=str.lower)

    expected_dtypes = {
        CONIFER: np.int8,
        DBH: np.float64,
        LATITUDE: np.float64,
        LONGITUDE: np.float64,
    }

    expected_columns = set(expected_dtypes.keys())
    columns = set(plot.columns)
    if expected_columns != columns.intersection(expected_columns):
        raise ValueError("Columns do not match expected columns")

    plot = plot.astype(expected_dtypes)

    return plot


@typechecked
def _compute_target(
    broadleafs: ee.FeatureCollection,
    conifers: ee.FeatureCollection,
    plot: pd.DataFrame,
    area_as_target: bool,
) -> Tuple[ee.Image, Tuple[ee.Geometry, float, str]]:
    # Get region of interest (ROI)
    roi = broadleafs.merge(conifers).geometry()

    # Get CRS in epsg format for center of the roi
    longitude, latitude = roi.centroid(1).getInfo()["coordinates"]
    crs = _sentinel_crs(latitude, longitude)

    # Convert ROI to bounds in output crs
    roi = roi.bounds(0.01, crs)

    # Check if rectangle has reasonable size
    roi_area = roi.area(0.01).getInfo()
    if roi_area == 0:
        raise ValueError(
            "Plot bounding box has area 0. Check if plot coordinates are valid."
        )
    if roi_area > 1e7:
        raise ValueError(
            "Plot bounding box has area > 1e7. Check if plot coordinates are valid."
        )

    # Render plot as fine resolution image, then reduce to coarse resolution
    fine_scale = min(plot["dbh"].min() * 5, SCALE)
    if plot["dbh"].min() < 0.05:
        print("Info: DBH < 0.05 m found. Google Earth Engine might ignore small trees.")
        fine_scale = 0.25

    # Compute broadleaf and conifer area
    broadleaf_area = _compute_area(broadleafs, SCALE, fine_scale, crs)
    conifer_area = _compute_area(conifers, SCALE, fine_scale, crs)

    # Compute target (conifer proportion) from broadleaf_area and conifer_area
    total_area = broadleaf_area.add(conifer_area)
    if area_as_target:
        target = broadleaf_area.addBands(conifer_area)
        target = target.updateMask(total_area.gt(0))  # Remove pixels with no trees
        target = target.rename([BROADLEAF_AREA, CONIFER_AREA])
    else:
        target = conifer_area.divide(total_area)
        target = target.updateMask(total_area.gt(0))  # Remove pixels with no trees
        target = target.rename(CONIFER_PROPORTION)

    return target, (roi, SCALE, crs)


@lru_cache
@typechecked
def list_reducers(use_buffered_reducers: bool = True) -> List[str]:
    """Lists all valid reducers in the Earth Engine API.

    Args:
        use_buffered_reducers:
            A boolean indicating whether to use the buffered reducers. If False the Google Earth Engine API is used to retrieve all current reducers (slow). Defaults to True.

    Returns:
        A list of strings representing the valid reducers.
    """
    if use_buffered_reducers:
        return [
            "And",
            "Or",
            "allNonZero",
            "anyNonZero",
            "bitwiseAnd",
            "bitwiseOr",
            "circularMean",
            "circularStddev",
            "circularVariance",
            "count",
            "countDistinct",
            "countDistinctNonNull",
            "countRuns",
            "first",
            "firstNonNull",
            "kendallsCorrelation",
            "kurtosis",
            "last",
            "lastNonNull",
            "max",
            "mean",
            "median",
            "min",
            "minMax",
            "mode",
            "product",
            "sampleStdDev",
            "sampleVariance",
            "skew",
            "stdDev",
            "sum",
            "variance",
        ]

    # Initialize Earth Engine API
    _initialize_ee()
    print("Checking for valid reducers...")

    # Create dummy image collection and point
    image = ee.Image.constant(0)
    collection = ee.ImageCollection(image)
    point = ee.Geometry.Point(0, 0)

    # Get all valid reducers
    attrs = dir(ee.Reducer)
    attrs.pop(attrs.index("reset"))
    reducers = []
    for attr in tqdm(attrs):
        attribute = getattr(ee.Reducer, attr)

        try:
            # Call without arguments
            reducer = attribute()

            # Apply reducer to image collection
            reduced_image = collection.reduce(reducer)

            # Execute computation
            reduced_point = reduced_image.reduceRegion(
                ee.Reducer.first(), geometry=point
            )
            reduced_point = reduced_point.getInfo()
        except (TypeError, ee.EEException):
            continue

        if all(v is None or isinstance(v, Number) for v in reduced_point.values()):
            reducers.append(attr)

    return reducers


@lru_cache
@typechecked
def list_reducer_bands(reducer: str) -> list[str] | None:
    """Lists all valid bands for a given reducer in the Earth Engine API.

    Args:
        reducer:
            A string representing the reducer. Run list_reducers() to get all valid reducer names.

    Returns:
        A list of strings representing the valid bands for the given reducer. None if the reducer does not return a multi-band image.
    """
    _initialize_ee()

    image = ee.Image.constant(0)
    collection = ee.ImageCollection(image)

    reduced = collection.reduce(getattr(ee.Reducer, reducer)())
    band_names = reduced.bandNames().getInfo()

    if len(band_names) == 1:
        return None

    reducer_bands = [band_name.split("_", 1)[1] for band_name in band_names]

    return reducer_bands


@lru_cache
@typechecked
def list_bands(level_2a: bool = True) -> List[str]:
    """Lists all valid Sentinel-2 bands offered by the Earth Engine API.

    Args:
        level_2a:
            A boolean indicating whether to list bands from Level-2A or Level-1C Sentinel-2 data. Level-2A if True. Defaults to True.

    Returns:
        A list of strings representing the valid Sentinel-2 bands.
    """
    # Initialize Earth Engine API
    _initialize_ee()

    # Get all valid bands
    if level_2a:
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    else:
        s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    bands = s2.first().bandNames().getInfo()

    return bands


@lru_cache
@typechecked
def list_indices() -> List[str]:
    """Lists all valid indices for Sentinel-2 offered by eemont.

    Returns:
        A list of strings representing the valid indices.
    """
    # Initialize Earth Engine API
    _initialize_ee()

    # Get all valid indices
    indices = eemont.common.indices()
    s2_indices = [
        index
        for index in indices
        if "Sentinel-2" in eemont.common.indices()[index]["platforms"]
    ]

    # Remove NIRvP (does need bands not available in Sentinel-2)
    s2_indices.remove("NIRvP")

    return s2_indices


@typechecked
def combine_band_name(
    composite_idx: int,
    band: str,
    reducer: str,
    reducer_band: str | None = None,
) -> str:
    """Combines a composite index, band label and reducer (+ reducer band) into a single band name.

    Args:
        composite_idx:
            An integer for the composite index starting at 1.
        band:
            A string for the band label in the format used by list_bands().
        reducer:
            A string for the reducer in the format used by list_reducers().
        reducer_band:
            A string for the reducer band in the format used by list_reducer_bands(). Defaults to None.

    Returns:
        A string for the band name.
    """
    # Create band name
    band_name = f"{composite_idx} {band} {reducer}"
    if reducer_band is not None:
        band_name += f" {reducer_band}"

    return band_name


@typechecked
def split_band_name(band_name: str) -> tuple[int, str, str, str | None]:
    """Splits a band name into its composite index, band label and reducer (+ reducer band).

    Args:
        A string for the band name. Expects format of combine_band_name().

    Returns:
        A tuple of the composite index starting at 1, band label and reducer (+ reducer band). The formatting of list_bands(), list_reducers() and list_reducer_bands() is used, assuming the band_name is in the format of combine_band_name().
    """
    # Split band name
    parts = band_name.split(" ")
    if not 3 <= len(parts) <= 4:
        raise ValueError("Band name does not have the expected format")

    # Extract parts
    composite_idx, band, reducer = parts[:3]
    reducer_band = None if len(parts) == 3 else parts[3]

    return int(composite_idx), band, reducer, reducer_band


@typechecked
def compute_label(
    target_path: str,
    plot: pd.DataFrame,
    area_as_target: bool = False,
) -> None:
    """Computes the label for a plot.

    The resulting raster has values between 0 and 1 for the conifer proportion. 1 being fully conifer and 0 being fully broadleaf for a given raster cell. If area_as_target is True, the raster will contain the area of broadleafs and conifers in square meters per raster cell.

    Args:
        target_path:
            A string representing the file path to save the raster. The suffix '.tif' (GeoTIFF) is expected.
        plot:
            A pandas DataFrame containing data on a single tree level with one column for longitude and latitude each, one column for DBH, and one for whether or not the tree is a conifer (1 is conifer, 0 is broadleaf). The column names must be 'longitude', 'latitude', 'dbh', and 'broadleaf' respectively.
        area_as_target:
            A boolean indicating whether to compute the area per leaf type instead of the leaf type mixture as labels. Results in a target with two bands, one for each leaf type. Defaults to False.
    """
    # Initialize Earth Engine API
    _initialize_ee()
    print("Preparing labels...")

    # Ensure proper plot DataFrame and path format
    plot = _check_plot(plot)
    _check_path(target_path, suffix=".tif", check_self=False)

    # Upload plot
    broadleafs = []
    conifers = []
    for _, row in plot.iterrows():
        circle = ee.Geometry.Point(
            [
                row["longitude"],
                row["latitude"],
            ]
        ).buffer(row["dbh"] / 2)

        if row["conifer"]:
            conifers.append(circle)
        else:
            broadleafs.append(circle)
    broadleafs = ee.FeatureCollection(broadleafs)
    conifers = ee.FeatureCollection(conifers)

    # Compute target
    target, (roi, scale, crs) = _compute_target(
        broadleafs, conifers, plot, area_as_target
    )

    # Clip to roi and reproject
    target = target.clip(roi)
    target = target.reproject(scale=scale, crs=crs)

    # Save target
    print("Computing labels...")
    _save_image(target, target_path)


@typechecked
def sentinel_composite(  # pylint: disable=too-many-arguments,too-many-locals
    target_path_from: str,
    data_path_to: str,
    time_window: Tuple[datetime.date, datetime.date],
    num_composites: int = 1,
    temporal_reducers: List[str] | None = None,
    indices: List[str] | None = None,
    level_2a: bool = True,
    sentinel_bands: List[str] | None = None,
    remove_clouds: bool = True,
    batch_size: int | None = None,
) -> None:
    """Creates a composite from many Sentinel-2 satellite images for a given label image.

    The raster will be saved to data_path_to after processing. The processing itself can take several minutes, depending on Google Earth Engine and the size of your region of interest. If you hit some limit of Google Earth Engine, the function will raise an error.

    Args:
        target_path_from:
            A string representing the file path to the label raster. This is used to derive the bounds, coordinate reference system and pixel size of the image. The suffix '.tif' (GeoTIFF) is expected.
        data_path_to:
            A string representing the output file path to save the composite raster. The suffix '.tif' (GeoTIFF) is expected.
        time_window:
            A tuple of two dates representing the start and end of the time window in which the satellite images are retrieved. The dates are converted to milliseconds. The end date is technically excluded, but only by one millisecond.
        num_composites:
            An integer representing the number of composites to create within the time window. The time window will be divided into equally long composite time windows. Defaults to 1.
        temporal_reducers:
            A list of strings representing the temporal reducers to use when creating the composite. Run list_reducers() or see https://developers.google.com/earth-engine/guides/reducers_intro for more information. None is replaced by ['mean']. Defaults to None.
        indices:
            A list of strings representing the spectral indices to add to the composite as additional bands. Run list_indices() or see https://eemont.readthedocs.io/en/latest/guide/spectralIndices.html for more information.
        level_2a:
            A boolean indicating whether to use Level-2A or Level-1C Sentinel-2 data. Defaults to True.
        sentinel_bands:
            A list of strings representing the bands to use from the Sentinel-2 data. For available bands run list_bands() or see https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED#bands (Level-1C) and https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED#bands (Level-2A). All bands are used if None. Defaults to None.
        remove_clouds:
            A boolean indicating whether to remove clouds from the satellite images based on the QA60 band. Uses MSK_CLDPRB=0 and MSK_SNWPRB=0 in addition for Level 2A images. Defaults to True.
        batch_size:
            An integer representing the number of bands used for one batch. All batches are process asynchronously. If None, all images are processed at once. Decrease the batch size if you hit computation limits of the Google Earth Engine. A smaller batch_size takes longer to compute. Defaults to None.
    """
    # Initialize Earth Engine API
    _initialize_ee()
    print("Preparing Sentinel-2 data...")

    # Check if band limit of 5000 is exceeded
    _check_band_limit(
        sentinel_bands,
        level_2a,
        indices,
        temporal_reducers,
        num_composites,
    )

    # Get region of interest (ROI), scale, and coordinate reference system (CRS)
    roi, scale, crs = _get_roi_scale_crs(target_path_from)

    # Get Sentinel-2 image collection filtered by bounds
    s2 = ee.ImageCollection(
        "COPERNICUS/S2_SR_HARMONIZED" if level_2a else "COPERNICUS/S2_HARMONIZED"
    )
    s2 = s2.filterBounds(roi)

    # Compute composite for each time windows
    data = []
    for start, end in _split_time_window(time_window, num_composites, level_2a):
        # Filter by roi and timewindow
        s2_window = s2.filterDate(
            round(start.timestamp() * 1000), round(end.timestamp() * 1000)
        )

        # Compute collection with selected bands and indices
        s2_window = _select_bands(
            s2_window,
            sentinel_bands,
            indices,
            level_2a,
            remove_clouds,
        )

        # Add to data
        data.append(_reduce_window(s2_window, temporal_reducers))

    # Stack images
    data = ee.ImageCollection(data).toBands()
    data = data.clip(roi)
    data = data.reproject(scale=scale, crs=crs)

    # Prettify band names
    data = _prettify_band_names(data)

    # Save data
    print("Computing data...")
    _save_image(data, data_path_to, batch_size=batch_size)


@typechecked
def shapefile2raster(
    shapefile_path: str,
    raster_path: str,
) -> None:
    """Creates a raster mask from a shapefile.

    Args:
        shapefile_path:
            A string representing the file path to the shapefile. The suffix '.shp' is expected.
        raster_path:
            A string representing the file path to save the raster mask. All cells partially or fully inside the shapefile will be 1 and all cells outside the shapefile will be NaN.  The suffix '.tif' (GeoTIFF) is expected.
    """
    # Initialize Earth Engine API
    _initialize_ee()
    print("Preparing labels...")

    # Ensure proper path format
    _check_path(shapefile_path, suffix=".shp")
    _check_path(raster_path, suffix=".tif", check_self=False)

    # Load shapefile
    shapefile = gpd.read_file(shapefile_path)
    shapefile = shapefile.to_crs("EPSG:4326")

    polygons = []
    for _, row in shapefile.iterrows():
        polygon = row.geometry
        geojson = polygon.__geo_interface__["coordinates"]
        gee_polygon = ee.Geometry.Polygon(geojson)
        polygons.append(gee_polygon)
    polygons = ee.FeatureCollection(polygons)

    # Get region of interest (ROI)
    roi = polygons.geometry()

    # Get CRS in epsg format for center of the roi
    longitude, latitude = roi.centroid(1).getInfo()["coordinates"]
    crs = _sentinel_crs(latitude, longitude)

    # Convert ROI to bounds in output crs
    roi = roi.bounds(0.01, crs)

    # Check if rectangle has reasonable size
    roi_area = roi.area(0.01).getInfo()
    if roi_area == 0:
        raise ValueError(
            "Plot bounding box has area 0. Check if plot coordinates are valid."
        )
    if roi_area > 1e10:
        raise ValueError(
            "Plot bounding box has area > 1e10. Check if plot coordinates are valid."
        )

    # Clip to roi and reproject
    target = ee.Image.constant(1)
    target = target.clip(polygons.geometry())
    target = target.reproject(scale=SCALE, crs=crs)

    # Save target
    print("Computing image...")
    _save_image(target, raster_path)


@typechecked
def download_dlt_2018(
    reference_path: str,
    destination_path: str,
) -> None:
    """Downloads the Dominant Leaf Type (DLT) 2018 for a given reference raster.

    The saved DLT raster is a raster with the values 0 and 1 representing the dominant leaf type broadleaf and conifer respectively. Unlabeled pixels are NaN. The raster is saved to destination_path.

    Args:
        reference_path:
            A string representing the file path to the reference raster. The suffix '.tif' (GeoTIFF) is expected.
        destination_path:
            A string representing the output file path to save the DLT raster. The suffix '.tif' (GeoTIFF) is expected.
    """
    with rasterio.open(reference_path) as src:
        crs = src.crs
        bounds = src.bounds
        width = src.width
        height = src.height

    url = "https://copernicus.discomap.eea.europa.eu/arcgis/services/GioLandPublic/HRL_DominanteLeafType_2018/ImageServer/WMSServer"

    params = {
        "request": "GetMap",
        "layers": "0",
        "format": "image/tiff",
        "width": width,
        "height": height,
        "bbox": ",".join(map(str, bounds)),
        "crs": crs,
    }

    url = url + "?" + "&".join([f"{k}={v}" for k, v in params.items()])

    with urlopen(url) as response:
        with MemoryFile(response.read()) as memfile:
            with memfile.open() as src:
                img = src.read(1).astype(float)
                img[(img < 1) | (img > 2)] = np.nan
                img = img - 1

                profile = src.profile
                profile["dtype"] = rasterio.float32
                profile["nodata"] = np.nan

    with rasterio.open(destination_path, "w", **profile) as dst:
        dst.write(img, 1)
        dst.descriptions = ["Conifer"]
