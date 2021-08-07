"""ExtraTrees (ET)"""

from skopt.space import Real, Categorical, Integer
from sklearn.ensemble import ExtraTreesClassifier

def get_instance():
    return ExtraTreesClassifier()

def get_name():
    return "Extra Trees"

def get_initials():
    return "ET"

def get_search_spaces():
    return {
        'n_estimators': Integer(150, 1100),
        'criterion': Categorical(['gini', 'entropy']),
        'max_depth': Integer(20, 150),
        'min_samples_split': Integer(2,20),
        'min_samples_leaf': Integer(1, 5),
        'max_features': Categorical(['auto', 'sqrt', 'log2']),
        'max_leaf_nodes': Integer(50, 150),
        'warm_start': [True, False],
        'max_samples': Real(0.1, 0.9)
    }