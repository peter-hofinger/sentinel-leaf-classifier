import unittest

import numpy as np
from optuna import Study
from skelm import ELMRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.pipeline import Pipeline

from ltm.models import cv_predict, hyperparam_search

from .test_features import fixture_data_path  # pylint: disable=unused-import
from .test_features import fixture_target_path  # pylint: disable=unused-import


class TestHyperparamSearch(unittest.TestCase):
    def setUp(self):
        regression = make_regression(n_samples=100, n_features=10, random_state=42)
        self.data, self.target = (  # pylint: disable=unbalanced-tuple-unpacking
            regression
        )

        def suggest_float(*args, **kwargs):
            return "suggest_float", args, kwargs

        def suggest_categorical(*args, **kwargs):
            return "suggest_categorical", args, kwargs

        self.model = ELMRegressor(random_state=42)
        self.search_space = [
            suggest_float("alpha", 1e-8, 1e5, log=True),
            suggest_categorical("include_original_features", [True, False]),
            suggest_float("n_neurons", 1, 1000),
            suggest_categorical("ufunc", ["tanh", "sigm", "relu", "lin"]),
            suggest_float("density", 0.01, 0.99),
        ]
        self.scorer = make_scorer(root_mean_squared_error, greater_is_better=False)
        self.refit = "mean_squared_error"

    def test_hyperparam_search(self):
        elm_model, elm_study = hyperparam_search(
            self.model,
            self.search_space,
            self.data,
            self.target,
            self.scorer,
            n_trials=1,
            random_state=42,
        )
        self.assertIsInstance(elm_model, Pipeline)
        self.assertIsInstance(elm_study, Study)
        self.assertIsInstance(elm_study.best_params, dict)
        self.assertLess(elm_study.best_value, -0.5)

    def test_reproducible_seed(self):
        elm_model1, elm_study1 = hyperparam_search(
            self.model,
            self.search_space,
            self.data,
            self.target,
            self.scorer,
            n_trials=1,
            random_state=42,
        )
        elm_model2, elm_study2 = hyperparam_search(
            self.model,
            self.search_space,
            self.data,
            self.target,
            self.scorer,
            n_trials=1,
            random_state=42,
        )
        self.assertEqual(elm_study1.best_params, elm_study2.best_params)
        self.assertEqual(elm_model1[-1].get_params(), elm_model2[-1].get_params())


def test_cv_predict(data_path, target_path):
    model = ELMRegressor(random_state=42)
    model.fit(np.random.rand(10 * 20, 30), np.random.rand(10 * 20))

    # Test whether same pixels are masked in raster than in target
    prediction = cv_predict(model, data_path, target_path, cv=2)

    assert np.isnan(prediction[0, 0, 0])
    assert prediction.shape == (1, 10, 20)
