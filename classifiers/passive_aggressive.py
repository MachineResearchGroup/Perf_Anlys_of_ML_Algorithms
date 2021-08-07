"""Passive Aggressive (PA)"""

from skopt.space import Real, Categorical, Integer
from sklearn.linear_model import PassiveAggressiveClassifier

def get_instance():
    return PassiveAggressiveClassifier()

def get_name():
    return "Passive Aggressive"

def get_initials():
    return "PA"

def get_search_spaces():
    return {
        'tol': Real(1e-5, 1e-3),
        'C': Real(1e-2, 1e1),
        'fit_intercept': [True, False],
        'max_iter': Integer(1000, 1500),
        'early_stopping': [True],
        'validation_fraction': [0.1, 0.2],
        'n_iter_no_change': [5, 10],
        'loss': ['hinge', 'squared_hinge'],
        'warm_start': [True, False],
        'n_jobs': [-1]
    }