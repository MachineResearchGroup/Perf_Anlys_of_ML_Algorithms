"""Multilayer Perceptron (MLP)"""

from skopt.space import Real, Categorical, Integer
from sklearn.neural_network import MLPClassifier

def get_instance():
    return MLPClassifier()

def get_name():
    return "Multilayer Perceptron"

def get_initials():
    return "MLP"

def get_search_spaces():
    return {
        'hidden_layer_sizes': Integer(20, 250),
        'activation': Categorical(['tanh', 'relu']),
        'solver': Categorical(['adam']),
        'batch_size': Integer(32, 480),
        'learning_rate_init': Real(1e-3, 0.1),
        'validation_fraction': [0.1, 0.2],
        'n_iter_no_change':[5, 10],
        'early_stopping': [True],
        'max_iter': Integer(20, 500),
        'tol': Real(1e-5, 1e-3),
        'warm_start': [True, False]
    }