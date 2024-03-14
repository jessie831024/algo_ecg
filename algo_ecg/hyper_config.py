
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import SelectFromModel
from algo_ecg.feature_transformer import AllFeatureCustomTransformer, RemoveCorrelatedFeatures
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold

my_splitter = StratifiedGroupKFold(n_splits=2)

hyperparameter_options = {
    'run_lr': {
        'my_pipe': Pipeline(steps=[
            ('features_all', AllFeatureCustomTransformer(axis=1)),
            ('feature_selection_corr', RemoveCorrelatedFeatures(threshold=0.9)),
            ('feature_selection', SelectFromModel(LinearSVC(dual="auto", penalty="l1", C=0.01,
                                                            class_weight="balanced", max_iter=10000))),
            ('standardscaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=1000, tol=0.1, class_weight="balanced"))
        ]),
        'my_param_grid': {
            "feature_selection_corr__threshold": (0.8, 0.9),
            "feature_selection__estimator__C": np.logspace(-2, 2, 2),
            "lr__C": np.logspace(-2, 2, 2)
        },
        'search_method': 'RandomizedSearchCV',
        'additional_params': {
            'n_iter': 2, 'cv': my_splitter, 'verbose': 10,
            'random_state': 42, 'n_jobs': -1
        }
    },
    'run_lr_halving': {
        'my_pipe': Pipeline(steps=[
            ('features_all', AllFeatureCustomTransformer(axis=1)),
            ('feature_selection_corr', RemoveCorrelatedFeatures(threshold=0.9)),
            ('feature_selection', SelectFromModel(LinearSVC(dual="auto", penalty="l1", C=0.01,
                                                            class_weight="balanced", max_iter=10000))),
            ('standardscaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=1000, tol=0.1, class_weight="balanced"))
        ]),
        'my_param_grid': {
            "feature_selection_corr__threshold": (0.8, 0.9),
            "feature_selection__estimator__C": np.logspace(-2, 2, 2),
            "lr__C": np.logspace(-2, 2, 2)
        },
        'search_method': 'HalvingRandomSearchCV',
        'additional_params': {
            'n_candidates': 10, 'aggressive_elimination': True,
            'cv': my_splitter, 'verbose': 10,
            'random_state': 42, 'n_jobs': -1
        }
    },
    'run_xgb': {
        'my_pipe': Pipeline(steps=[
            ('features_all', AllFeatureCustomTransformer(axis=1)),
            ('feature_selection_corr', RemoveCorrelatedFeatures(threshold=0.9)),
            ('feature_selection', SelectFromModel(LinearSVC(dual="auto", penalty="l1", C=0.01,
                                                            class_weight="balanced", max_iter=10000))),
            ('standardscaler', StandardScaler()),
            ("xgb", XGBClassifier())]
        ),
        'my_param_grid': {
            "feature_selection_corr__threshold": (0.7, 0.8, 0.9),
            "feature_selection__estimator__C": np.logspace(-3, 3, 3),
            'xgb__learning_rate': [0.1, 0.3, 0.5],
            'xgb__max_depth': [3, 5, 7],
            'xgb__n_estimators': [50, 100, 200]
        },
        'search_method':  'HalvingRandomSearchCV',
        'additional_params': {
            'n_candidates': 5, 'aggressive_elimination': True,
            'cv': my_splitter, 'verbose': 10,
            'random_state': 42, 'n_jobs': -1
        }
    },
    'run_svc': {
        'my_pipe': Pipeline(steps=[
            ('features_all', AllFeatureCustomTransformer(axis=1)),
            ('feature_selection_corr', RemoveCorrelatedFeatures(threshold=0.9)),
            ('feature_selection', SelectFromModel(LinearSVC(dual="auto", penalty="l1", C=0.01,
                                                            class_weight="balanced", max_iter=10000))),
            ('standardscaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', probability=True))]
        ),
        'my_param_grid': {
            "feature_selection_corr__threshold": (0.7, 0.8, 0.9),
            "feature_selection__estimator__C": np.logspace(-3, 3, 3),
            "svc__C": np.logspace(-2, 10, 10),
            "svc__gamma": np.logspace(-9, 3, 10)
        },
        'search_method': 'RandomizedSearchCV',
        'additional_params': {
            'n_iter': 10, 'cv': my_splitter, 'verbose': 10,
            'random_state': 42, 'n_jobs': -1}
    }
}








