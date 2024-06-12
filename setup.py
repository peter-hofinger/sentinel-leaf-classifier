# pylint: disable=missing-module-docstring
from setuptools import find_packages, setup

setup(
    name="ltm",
    packages=find_packages(),
    version="1.0.0",
    description="Predicting leaf type mixture using tailored sentinel-2 composites.",
    author="Peter Hofinger",
    license="MIT",
    install_requires=[
        "aiohttp",
        "dask[dataframe,distributed]",  # for scikit-elm
        "dill",  # for pickling objects with lambda functions
        "ee",
        "eemont",
        "geopandas",
        "ipywidgets",  # for tqdm
        "rasterio",
        "matplotlib",
        "nest_asyncio",
        "numpy",
        "optuna",
        "pandas",
        "pyproj",
        "requests",
        "scikit-elm",
        "scikit-learn",
        "SciencePlots",
        "seaborn",
        "tqdm",
        "typeguard==4.2.1",  # stable version
        "utm",
        "xgboost",
    ],
    python_requires=">=3.10",
    extras_require={
        "linting": [
            "pylint",
            "pytest",
        ],
        "testing": [
            "pytest",
        ],
    },
)
