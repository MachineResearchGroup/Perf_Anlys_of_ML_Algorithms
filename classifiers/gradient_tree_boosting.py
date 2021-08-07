"""Gradient Tree Boosting (GTB)"""

from skopt.space import Real, Categorical, Integer
from sklearn.ensemble import GradientBoostingClassifier

def get_instance():
    return GradientBoostingClassifier()

def get_name():
    return "Gradient Tree Boosting"

def get_initials():
    return "GTB"

def get_search_spaces():
    return {
        'loss': ['deviance'],
        'learning_rate': Real(1e-1, 1e1),
        'n_estimators': Integer(100, 1000),
        'criterion': Categorical(['friedman_mse', 'mse', 'mae']),
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(1, 5),
        'max_depth': Integer(20, 150),
        'max_features': Categorical(['auto', 'sqrt', 'log2']),
        'max_leaf_nodes': Integer(50, 150),
        'warm_start': [True, False],
        'validation_fraction': [0.1, 0.2],
        'n_iter_no_change': Integer(5, 10),
        'tol': Real(1e-5, 1e-3)
    }