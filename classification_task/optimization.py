from tools import time
from tools import compute
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
__n_iter = 150
__n_splits = 10
__test_size = 0.2
__train_size = 0.8
def get_optimized_model(classifier, x, y, fold):
    
    cv = StratifiedShuffleSplit(n_splits=__n_splits, test_size=__test_size)
    search_spaces = classifier.get_search_spaces()
    estimator = classifier.get_instance()
    model_name = classifier.get_name()

    print("\nOptimizing " + model_name + " algorithm...")

    model = BayesSearchCV(estimator=estimator, search_spaces=search_spaces, 
    n_iter=__n_iter, scoring='f1_macro', cv=cv, refit=True, return_train_score=True, 
    n_jobs=3, n_points=3, pre_dispatch=3)
    optimization_time = time
    optimization_time.init()
    model.fit(x, y)
    optimization_time.end()
    compute.hyper_results(model, model_name, fold, optimization_time)
    return model.best_estimator_