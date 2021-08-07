"""Stochastic Gradient Descent (SGD)"""

from skopt.space import Real, Categorical, Integer
from sklearn.linear_model import SGDClassifier

def get_instance():
    return SGDClassifier()

def get_name():
    return "Stochastic Gradient Descent"

def get_initials():
    return "SGD"

def get_search_spaces():
    return {
        'loss': Categorical(['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']),
        'penalty': Categorical(['l1', 'l2', 'elasticnet']),
        'alpha': Real(1e-4, 1e-2),
        'l1_ratio': Real(0, 1),
        'fit_intercept': [True, False],
        'max_iter': Integer(500, 1500),
        'tol': Real(1e-5, 1e-3),
        'shuffle': [True, False],
        'epsilon': Real(1e-2, 1) ,
        'n_jobs': [-1],
        'learning_rate': Categorical(['optimal', 'invscaling', 'adaptive']),
        'eta0': Real(1e-2, 1e1),
        'power_t': Real(0, 0.1),
        'early_stopping': [True],
        'validation_fraction': [0.1, 0.2],
        'n_iter_no_change': [5, 10],
        'warm_start': [True, False],
        'average': [True, False],
    }