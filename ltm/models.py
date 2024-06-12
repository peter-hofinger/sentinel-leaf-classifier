"""Performs hyperparameter search using Optuna and offers cross-validated inference on raster data.

Typical usage example:

    from ltm.features import load_raster
    from ltm.models import hyperparam_search
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import make_scorer, root_mean_squared_error

    def suggest_categorical(*args, **kwargs):
        return "suggest_categorical", args, kwargs

    model = RandomForestRegressor()
    search_space = [suggest_categorical("n_estimators", [100, 200, 300])]
    scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

    data = load_raster("data_interpolated.tif")
    target = load_raster("target.tif")
    data, target = data[target.notna()], target[target.notna()]

    best_model, study = hyperparam_search(
        model,
        search_space,
        data,
        target,
        scorer,
    )
"""

from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Any, Callable, Dict, Iterator, List, Tuple

import dill
import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
import rasterio
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.metrics._scorer import _BaseScorer
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm
from typeguard import typechecked

from ltm.data import (
    BROADLEAF_AREA,
    CONIFER_AREA,
    list_bands,
    list_indices,
    sentinel_composite,
)
from ltm.features import (
    interpolate_data,
    load_raster,
    np2pd_like,
    save_raster,
    to_float32,
)


@typechecked
class EndMemberSplitter(BaseCrossValidator):
    """K-fold splitter that only uses end members for training.

    End members are defined as instances with label 0 or 1. Using this option with area per leaf type as labels is experimental.

    Attributes:
        n_splits:
            Number of splits to use for kfold splitting.
        k_fold:
            KFold object to use for splitting.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        """Initializes the EndMemberSplitter.

        Args:
            n_splits:
                Number of splits to use for kfold splitting.
            shuffle:
                Whether to shuffle the data before splitting into batches. Defaults to False.
            random_state:
                Random state to use for reproducible results.
        """
        self.n_splits = n_splits
        self.k_fold = KFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def _iter_test_indices(
        self,
        X: npt.ArrayLike | None = None,
        y: npt.ArrayLike | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[np.ndarray]:
        """Generates integer indices corresponding to test sets.

        Args:
            X:
                Data to split.
            y:
                Labels to split.
            groups:
                Group labels to split.

        Yields:
            Integer indices corresponding to test sets.
        """
        fun = self.k_fold._iter_test_indices  # pylint: disable=protected-access
        for test in fun(X, y, groups):
            if y is not None:
                indices = np.where((y[test] == 0) | (y[test] == 1))[0]
                test = test[indices]
            yield test

    def split(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generates indices to split data into training and test set.

        Args:
            data:
                Data to split.
            target:
                Labels to split.
            groups:
                Group labels to split. Not used.

        Yields:
            Tuple of indices for training and test set.
        """
        X = np.array(X)
        y = np.array(y)
        for train, test in self.k_fold.split(X, y):
            indices = np.where((y[train] == 0) | (y[train] == 1))[0]

            end_member_train = train[indices]

            if end_member_train.shape[0] == 0:
                raise ValueError(
                    "No end members in one training set. Maybe you are just unlucky, try another random state."
                )

            yield end_member_train, test

    def get_n_splits(
        self,
        X: npt.ArrayLike | None = None,
        y: npt.ArrayLike | None = None,
        groups: np.ndarray | None = None,
    ) -> int:
        """Returns the number of splitting iterations in the cross-validator.

        Args:
            X:
                Data to split. Not used.
            y:
                Target to split. Not used.
            groups:
                Group labels to split. Not used.

        Returns:
            Number of splitting iterations in the cross-validator.
        """
        return self.n_splits


@typechecked
def _target2raster(
    target: pd.Series | pd.DataFrame,
    indices: np.ndarray,
    plot_shape: Tuple[int, int, int],
    area2mixture: bool = False,
) -> np.ndarray:
    # Create target values array of shape (n_samples, n_features)
    target_values = target.values
    if len(target_values.shape) == 1:
        target_values = np.expand_dims(target_values, axis=1)

    # Create raster from target with the help of an indices array
    raster_shape = (plot_shape[1] * plot_shape[2], plot_shape[0])
    raster = np.full(raster_shape, np.nan)
    raster[indices] = target_values
    raster = raster.reshape(plot_shape[1], plot_shape[2], plot_shape[0]).transpose(
        2, 0, 1
    )

    # Use indices of BROADLEAF_AREA and CONIFER_AREA to compute mixture = broadleaf / (broadleaf + conifer)
    if area2mixture:
        columns = list(target.columns)
        broadleaf_index = columns.index(BROADLEAF_AREA)
        conifer_index = columns.index(CONIFER_AREA)
        broadleaf = raster[broadleaf_index, :, :]
        conifer = raster[conifer_index, :, :]
        raster = broadleaf / (broadleaf + conifer)
        raster = np.expand_dims(raster, axis=0)

    return raster


@typechecked
def _has_nan_error(
    estimator: BaseEstimator,
) -> bool:
    # Checks if estimator raises a ValueError when predicting on NaN
    try:
        estimator.fit([[0]], [0])
        estimator.predict([[np.nan]])
        return False
    except ValueError:
        return True


@typechecked
def _build_pipeline(
    model: BaseEstimator,
    do_standardize: bool,
    do_pca: bool,
    n_components: int | None,
    model_params: Dict[str, Any],
) -> Pipeline:
    # Set params first, as it can change _has_nan_error
    model = model.set_params(**model_params)

    steps = []
    if _has_nan_error(model) or do_pca:
        steps.append(("imputer", KNNImputer()))
    if do_standardize:
        steps.append(("scaler", StandardScaler()))
    if do_pca:
        if n_components is None:
            raise ValueError("n_components must be set if do_pca is True.")
        steps.append(("pca", PCA(n_components=n_components)))

    steps.append(("model", model))

    return Pipeline(steps=steps)


@typechecked
def _study2model(
    study: optuna.study.Study,
    model: BaseEstimator,
    data: npt.ArrayLike,
    target: npt.ArrayLike,
) -> Pipeline:
    # Define preprocessing steps for best model
    model_params = {
        param: value
        for param, value in study.best_params.items()
        if param not in ["do_standardize", "do_pca", "n_components"]
    }

    n_components = None
    if study.best_params["do_pca"]:
        n_components = study.best_params["n_components"]

    best_model = _build_pipeline(
        model,
        study.best_params["do_standardize"],
        study.best_params["do_pca"],
        n_components,
        model_params,
    )

    # Fit best model
    best_model.fit(data, target)

    return best_model


@typechecked
def _create_paths(
    model: BaseEstimator,
    save_folder: str,
) -> Tuple[Path, Path, Path]:
    save_path_obj = Path(save_folder)
    if not save_path_obj.parent.exists():
        raise ValueError(
            f"Directory of save_path does not exist: {save_path_obj.parent}"
        )

    # Check if files already exist
    model_name = model.__class__.__name__
    cache_path = save_path_obj / f"{model_name}_cache.pkl"
    study_path = save_path_obj / f"{model_name}_study.pkl"
    model_path = save_path_obj / f"{model_name}.pkl"

    return study_path, model_path, cache_path


@typechecked
def _check_save_folder(
    model: BaseEstimator,
    data: npt.ArrayLike,
    target: npt.ArrayLike,
    save_folder: str | None,
    use_caching: bool,
) -> Tuple[Pipeline, optuna.study.Study] | None:
    # Check for valid save_path
    if save_folder is not None:
        study_path, model_path, _ = _create_paths(model, save_folder)

        if study_path.exists() and model_path.exists():
            print(f"Files already exist, skipping search: {study_path}, {model_path}")

            # Load best model and study
            with open(model_path, "rb") as file:
                best_model = dill.load(file)
            with open(study_path, "rb") as file:
                study = dill.load(file)

            return best_model, study

        if study_path.exists():
            # Inform user
            print("Creating model from study file...")

            # Load the study and create the best model
            with open(study_path, "rb") as file:
                study = dill.load(file)
            best_model = _study2model(study, model, data, target)

            # Save best model
            with open(model_path, "wb") as file:
                dill.dump(best_model, file)

            return best_model, study

        if model_path.exists():
            # Raise error if model file exists but study file is missing
            raise ValueError(
                f"Study file is missing, please delete the model file manually and rerun the script: {model_path}"
            )
    elif use_caching:
        print("Warning: use_caching=True but save_folder=None, caching is disabled.")

    return None


@typechecked
def _save_study_model(
    study: optuna.study.Study,
    best_model: Pipeline,
    study_path: Path,
    model_path: Path,
) -> None:
    # Save study
    with open(study_path, "wb") as file:
        dill.dump(study, file)

    # Save best model
    with open(model_path, "wb") as file:
        dill.dump(best_model, file)


@typechecked
def _get_composite_params() -> Dict[str, int]:
    compositing_path = "../reports/compositing.csv"

    try:
        df = pd.read_csv(compositing_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"File {compositing_path} not found. Please run the compositing notebook first."
        ) from exc

    metric = "Root Mean Squared Error"

    optimal_idx = df.groupby("Reducer")[metric].idxmin()
    optimal_df = df.loc[optimal_idx]
    optimal_df = optimal_df.set_index("Reducer")

    composite_dict = optimal_df["Composites"].to_dict()

    # Sort alphabetically
    composite_dict = dict(sorted(composite_dict.items(), key=lambda x: x[0]))

    return composite_dict


@typechecked
def _create_composites(
    year: int,
    data_folder: str,
    target_path: str,
    batch_size: int | None = None,
) -> None:
    # Get selected bands and indices
    importance_path = "../reports/band_importance.csv"
    if Path(importance_path).exists():
        sentinel_bands, indices = bands_from_importance(importance_path)
    else:
        raise FileNotFoundError(
            f"File {importance_path} not found. Please run the band importance notebook first."
        )

    # Get composites by most composites first
    composite_dict = _get_composite_params()
    composite_dict = dict(
        sorted(composite_dict.items(), key=lambda item: item[1], reverse=True)
    )

    # Create one composite for each reducer
    iterable = tqdm(composite_dict.items(), desc=f"Downloading Composites for {year}")
    for reducer, num_composites in iterable:
        composite_path = str(
            Path(data_folder) / f"{year}/data_{reducer}_{num_composites}.tif"
        )

        sleep_time = 60
        while not Path(composite_path).exists():
            try:
                sentinel_composite(
                    target_path,
                    composite_path,
                    time_window=(
                        datetime(year, 4, 1),
                        datetime(year + 1, 4, 1),
                    ),
                    num_composites=num_composites,
                    temporal_reducers=[reducer],
                    indices=indices,
                    sentinel_bands=sentinel_bands,
                    batch_size=batch_size,
                )
            except BaseException as e:  # pylint: disable=broad-exception-caught
                print(e)
                sleep(sleep_time)
                sleep_time *= 2


@typechecked
def bands_from_importance(
    band_importance_path: str,
    top_n: int = 30,
    level_2a: bool = True,
) -> Tuple[List[str], List[str]]:
    """Extracts the band names of Sentinel-2 bands and indices from the band importance file.

    The top_n bands are read from the band importance file. Those bands are then divided into Sentinel-2 bands and indices.

    Args:
        band_importance_path:
            Path to the file with band names and their scores. The bands are expected to be in reverse order of when they were removed by RFE. Only the band names and their order are used.
        top_n:
            Number of bands to keep. Defaults to 30.
        level_2a:
            Whether the band importance file is from a level 2A dataset. This is necessary for distinguishing Sentinel-2 bands from indices. Defaults to True.

    Returns:
        Tuple of lists of Sentinel-2 band names and index names as strings.
    """
    # Check path
    if not Path(band_importance_path).exists():
        raise ValueError(f"File does not exist: {band_importance_path}")

    # Read band importance file
    df = pd.read_csv(band_importance_path, index_col=0)
    band_names = df.index
    df = df.reset_index()

    # Divide bands into Sentinel-2 bands and indices
    best_bands = list(band_names[:top_n])
    valid_sentinel_bands = list_bands(level_2a)
    valid_index_bands = list_indices()
    sentinel_bands = [band for band in valid_sentinel_bands if band in best_bands]
    index_bands = [band for band in valid_index_bands if band in best_bands]

    # Sanity check
    if len(sentinel_bands) + len(index_bands) != len(best_bands):
        raise ValueError(
            "The sum of Sentinel-2 bands and index bands does not equal the number of best bands. This should not happen..."
        )

    return sentinel_bands, index_bands


@typechecked
def area2mixture_scorer(scorer: _BaseScorer) -> _BaseScorer:
    """Modifies the score function of a scorer to use the leaf type mixture calculated from leaf type areas.

    Args:
        scorer:
            A _BaseScorer scorer, e.g. returned by make_scorer().

    Returns:
        Scorer with modified score function.
    """
    score_func = scorer._score_func  # pylint: disable=protected-access

    def mixture_score_func(
        target_true: npt.ArrayLike,
        target_pred: npt.ArrayLike,
        *args,
        **kwargs,
    ) -> Callable:
        # Convert to np.ndarray
        target_true = np.array(target_true)
        target_pred = np.array(target_pred)

        # broadleaf is 0, conifer is 1
        target_true = target_true[:, 0] / (target_true[:, 0] + target_true[:, 1])
        target_pred = target_pred[:, 0] / (target_pred[:, 0] + target_pred[:, 1])

        return score_func(target_true, target_pred, *args, **kwargs)

    scorer._score_func = mixture_score_func  # pylint: disable=protected-access

    return scorer


@typechecked
def hyperparam_search(  # pylint: disable=too-many-arguments,too-many-locals
    model: BaseEstimator,
    search_space: List[Tuple[str, Tuple, Dict[str, Any]]],
    data: npt.ArrayLike,
    target: npt.ArrayLike,
    scorer: _BaseScorer,
    cv: int | BaseCrossValidator = 5,
    n_trials: int = 100,
    n_jobs: int = 1,
    random_state: int | None = None,
    save_folder: str | None = None,
    use_caching: bool = True,
    always_standardize: bool = False,
) -> Tuple[Pipeline, optuna.study.Study]:
    """Performs hyperparameter search for a model using Optuna.

    The search space will be explored using Optuna's TPE sampler, together with standardization and PCA for preprocessing. The best pipeline object will be returned along with the Optuna study. A KNNImputer is used for estimators not supporting NaN values.

    Args:
        model:
            Model to perform hyperparameter search for.
        search_space:
            List of tuples with the name of the method to suggest, the arguments and the keyword arguments. For example [("suggest_float", ("alpha", 1e-10, 1e-1), {"log": True})].
        data:
            Features to use for hyperparameter search.
        target:
            Labels to use for hyperparameter search.
        scorer:
            Scorer to use for hyperparameter search. Please make sure to set greater_is_better=False when using make_scorer if you want to minimize a metric.
        cv:
            Number of folds or BaseCrossValidator instance to use for cross validation. Defaults to 5.
        n_trials:
            Number of trials to perform hyperparameter search. Defaults to 100.
        n_jobs:
            Number of jobs to use for hyperparameter search. Set it to -1 to maximize parallelization.  Defaults to 1, as otherwise Optuna becomes non-deterministic.
        random_state:
            Integer to be used as random state for reproducible results. Defaults to None.
        save_folder:
            Folder to save the study PKL and model PKL to. Uses model.__class__.__name__ to name the files. Skips search if files already exist. Defaults to None.
        use_caching:
            Whether to use caching for the search. Saves a [model]_cache.pkl for each step and resumes if a cache exists. The cache is deleted after the final study is saved. Defaults to True.
        always_standardize:
            If true the data is always standardized. Recommended for SVM based estimators. Defaults to False.

    Returns:
        Tuple of the best pipeline and the Optuna study.
    """
    result = _check_save_folder(
        model,
        data,
        target,
        save_folder,
        use_caching,
    )
    if result is not None:
        return result

    # Create paths
    if save_folder is not None:
        study_path, model_path, cache_path = _create_paths(model, save_folder)

    def callback(study, _):
        # Save intermediate study
        if use_caching and save_folder is not None:
            with open(
                cache_path,  # pylint: disable=possibly-used-before-assignment
                "wb",
            ) as file:
                dill.dump(study, file)

    def objective(trial):
        # Choose whether to standardize and apply PCA
        standardize_options = [True, False]
        if always_standardize:
            standardize_options = [True]
        do_standardize = trial.suggest_categorical(
            "do_standardize", standardize_options
        )
        do_pca = trial.suggest_categorical("do_pca", [True, False])

        # Build pipeline
        n_components = None
        if do_pca:
            n_splits = cv if isinstance(cv, int) else cv.get_n_splits()
            max_components = min(data.shape) - np.ceil(
                min(data.shape) / n_splits
            ).astype(int)
            n_components = trial.suggest_int("n_components", 1, max_components)

        params = {
            args[0]: getattr(trial, name)(*args, **kwargs)
            for name, args, kwargs in search_space
        }

        nonlocal model
        pipe = _build_pipeline(model, do_standardize, do_pca, n_components, params)

        # Cross validate pipeline
        try:
            cv_results = cross_validate(
                pipe, data, target, cv=cv, scoring=scorer, n_jobs=-1
            )
            score = cv_results["test_score"].mean()

            return score
        # Catch case that all fits fail
        except ValueError:
            print("All fits failed, returning NaN.")
            return np.nan

    if use_caching and save_folder is not None and Path(cache_path).exists():
        # Resume search from cache
        with open(cache_path, "rb") as file:
            study = dill.load(file)
        n_trials -= len(study.trials)

        print(f"Resuming search from cache at trial {len(study.trials)}.")
    else:
        # Start new search
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=random_state),
            direction="maximize",
            study_name=model.__class__.__name__,
        )

    # Optimize study
    study.optimize(objective, callbacks=[callback], n_trials=n_trials, n_jobs=n_jobs)

    best_model = _study2model(study, model, data, target)

    if save_folder is not None:
        _save_study_model(study, best_model, study_path, model_path)

        # Delete cache
        if use_caching:
            Path(cache_path).unlink()

    return best_model, study


@typechecked
def cv_predict(
    model: BaseEstimator,
    data_path: str,
    target_path: str,
    cv: int | BaseCrossValidator | None = None,
) -> np.ndarray:
    """Predicts on rasters using cross_val_predict.

    Args:
        model:
            Regressor to use for prediction.
        data_path:
            A string with path to the data in GeoTIFF format.
        target_path:
            A string with path to the target data in GeoTIFF format.
        cv:
            An integer for the number of folds or a BaseCrossValidator object for performing cross validation. Will be passed to sklearn.model_selection.cross_val_predict(). Defaults to None.

    Returns:
        A numpy raster of the prediction in the format of read() from rasterio.
    """
    # Load data and plot shape
    data = load_raster(data_path)
    target = load_raster(target_path)

    with rasterio.open(target_path) as src:
        shape = src.read().shape

    # Remove NaNs while keeping the same indices
    indices_array = np.arange(shape[1] * shape[2])
    mask = target.notna()
    data, target, indices_array = data[mask], target[mask], indices_array[mask]

    # Predict using cross_val_predict
    target_pred = cross_val_predict(model, data, target, cv=cv, n_jobs=-1)
    target_pred = np2pd_like(target_pred, target)

    plot = _target2raster(target_pred, indices_array, shape)

    return plot


@typechecked
def create_data(
    year: int,
    target_path: str,
    data_folder: str,
    batch_size: int | None = None,
) -> None:
    """Creates a data raster for a given year ready to be used for inference using the Earth Engine API.

    The files band_importance.csv and compositing.csv are expected to be located in "../reports/". The data raster is saved in the data_folder/year folder with the final data raster being saved as data_folder/year/data.tif.

    Args:
        year:
            Year to create data for.
        target_path:
            Path to the target data in GeoTIFF format.
        data_folder:
            Path to the folder to save the data and composite rasters in.
        batch_size:
            Number of samples to download at a time. Defaults to None.
    """
    # Skip if data already exists
    data_path = Path(data_folder) / f"{year}/data.tif"
    if data_path.exists():
        return
    data_path.parent.mkdir(parents=True, exist_ok=True)

    # Create composites
    _create_composites(year, data_folder, target_path, batch_size)

    # Combine into one raster
    total_data = pd.DataFrame()
    composite_dict = _get_composite_params()
    for reducer, num_composites in tqdm(
        composite_dict.items(), desc=f"Combining Composites for {year}"
    ):
        composite_path = str(
            Path(data_folder) / f"{year}/data_{reducer}_{num_composites}.tif"
        )

        data = load_raster(composite_path)
        data = interpolate_data(data)
        data = to_float32(data)
        total_data = pd.concat([total_data, data], axis=1)

    # Save the concatenated data
    save_raster(total_data, target_path, str(data_path))
