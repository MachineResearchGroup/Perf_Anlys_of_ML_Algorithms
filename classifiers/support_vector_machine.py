"""Support Vector Machine (SVM)"""

from skopt.space import Real, Categorical, Integer
from sklearn.svm import SVC

def get_instance():
    return SVC()

def get_name():
    return "Support Vector Machine"

def get_initials():
    return "SVM"

def get_search_spaces():
    return {
        'C': Real(1e-2, 1e1),
        'kernel': Categorical(['linear', 'rbf']),
        'degree': Integer(2, 5),
        'gamma': Categorical(['scale', 'auto']),
        'shrinking': [True, False],
        'probability': [True, False],
        'tol': Real(1e-5, 1e-3),
        'cache_size': [500],
        'decision_function_shape': Categorical(['ovo', 'ovr']),
    }