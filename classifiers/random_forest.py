"""Random Forest (RF)"""

from skopt.space import Real, Categorical, Integer
from sklearn.ensemble import RandomForestClassifier

def get_instance():
    return RandomForestClassifier()

def get_name():
    return "Random Forest"

def get_initials():
    return "RF"

def get_search_spaces():
    return {
        'n_estimators': Integer(2, 1000),
        'criterion': Categorical(['gini', 'entropy']),
        'max_depth': Integer(20, 150),
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(1, 5),
        'min_weight_fraction_leaf': Real(0.0, 0.5),
        'max_features': Categorical(['auto', 'sqrt', 'log2']),
        'max_leaf_nodes': Integer(50, 150),
        'n_jobs': [-1],
        'warm_start': [True, False],
        'max_samples': Real(0.1, 0.9)
    }