"""Perceptron (P)"""

from skopt.space import Real, Categorical, Integer
from sklearn.linear_model import Perceptron

def get_instance():
    return Perceptron()

def get_name():
    return "Perceptron"

def get_initials():
    return "P"

def get_search_spaces():
    return {
        'penalty': Categorical(['l1', 'l2', 'elasticnet']),
        'alpha': Real(1e-3, 1e1),
        'fit_intercept': [True, False],
        'max_iter': Integer(500, 1500),
        'tol': Real(1e-5, 1e-3),
        'shuffle': [True, False],
        'eta0': Real(1e-1, 1e1),
        'n_jobs': [-1],
        'early_stopping': [True],
        'validation_fraction': [0.1, 0.2],
        'n_iter_no_change':[5, 10],
        'warm_start': [True, False]
    }
    