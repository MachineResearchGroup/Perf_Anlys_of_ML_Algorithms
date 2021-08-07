"""Logistic Regression (LR)"""

from skopt.space import Real, Categorical, Integer
from sklearn.linear_model import LogisticRegression

def get_instance():
    return LogisticRegression()

def get_name():
    return "Logistic Regression"

def get_initials():
    return "LR"

def get_search_spaces():
    return [{
        'penalty': Categorical(['l2', 'none']),
        'tol': Real(1e-5, 1e-3),
        'C': Real(1e-2, 1e1),
        'fit_intercept': [True, False],
        'solver': Categorical(['newton-cg', 'lbfgs', 'sag', 'saga']),
        'max_iter': Integer(1000, 2000),
        'multi_class': ['auto'],
        'warm_start': [True, False],
        'n_jobs': [-1]
    },{
        'penalty': ['elasticnet'],
        'tol': Real(1e-5, 1e-3),
        'C': Real(1e-2, 1e1),
        'fit_intercept': [True, False],
        'solver': ['saga'],
        'max_iter': Integer(1000, 2000),
        'multi_class': ['auto'],
        'warm_start': [True, False],
        'n_jobs': [-1],
        'l1_ratio': Real(0, 1)

    },{
        'penalty': ['l1'],
        'tol': Real(1e-5, 1e-3),
        'C': Real(1e-2, 1e1),
        'intercept_scaling': Real(1e-1, 1e1),
        'solver': ['liblinear', 'saga'],
        'max_iter': Integer(1000, 2000),
        'multi_class': ['ovr'],
        'warm_start': [True, False]
    },{
        'penalty': ['l2'],
        'dual': [True],
        'tol': Real(1e-5, 1e-3),
        'C': Real(1e-2, 1e1),
        'fit_intercept': [True, False],
        'solver': ['liblinear'],
        'max_iter': Integer(1000, 2000),
        'multi_class': ['auto'],
    }]