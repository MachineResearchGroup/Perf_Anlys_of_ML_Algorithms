"""k-Nearest Neighbors (KNN)"""

from skopt.space import Real, Categorical, Integer
from sklearn.neighbors import KNeighborsClassifier

def get_instance():
    return KNeighborsClassifier()

def get_name():
    return "k-Nearest Neighbors"

def get_initials():
    return "KNN"

def get_search_spaces():
    return {
        'n_neighbors': Integer(2, 10),
        'weights': Categorical(['uniform', 'distance']),
        'algorithm': Categorical(['brute']),
        'leaf_size': Integer(10, 60),
        'p': Integer(1, 2),
        'metric': Categorical(['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']),
        'n_jobs': [-1]
    }