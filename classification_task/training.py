from tools import data
from tools import time
from tools import compute
from classification_task import optimization
from sklearn.model_selection import StratifiedShuffleSplit
from classifiers import extra_trees, gradient_tree_boosting, \
    k_nearest_neighbors, logistic_regression, multilayer_perceptron, \
        multinomial_naive_bayes, passive_aggressive, random_forest, \
            stochastic_gradient_descent, support_vector_machine

__classifiers = [
    extra_trees, 
    gradient_tree_boosting, 
    k_nearest_neighbors, 
    logistic_regression, 
    multilayer_perceptron, 
    multinomial_naive_bayes,
    passive_aggressive,
    random_forest, 
    stochastic_gradient_descent,
    support_vector_machine 
    ]

def run(k_folds, inter):
    print("\nTraining the algorothms...")
    
    cv = StratifiedShuffleSplit(n_splits=k_folds, test_size=0.2)
    x = data.get_encoded_requirements("tfidf_requirements.npz")
    y = data.get_encoded_classes("tfidf_classes.npy")
    indexes = cv.split(x, y)
    
    for (train, test), fold in zip(indexes, range(k_folds)):
        print("\nfold "+str(fold+1))
        for classifier in __classifiers:
            model_time = time
            model_time.init()
            model_name = classifier.get_name()
            
            print('\nStart of the ' + model_name +
            ' algorithm at ' + model_time.get_init_date_time())
            
            model = optimization.get_optimized_model(classifier, x[train],y[train], fold+1)
            print("\nPredicting the test set...")
            predict = model.predict(x[test])

            model_time.end()

            compute.classification_results(y[test], predict, classifier, model_time, fold+1, inter)

            print('\nEnd of the ' + model_name + ' algorithm execution at ' +
                model_time.get_end_date_time() + '\nTotal run time: ' + 
                model_time.get_execu_time())

    compute.mean_per_interaction(inter, k_folds)