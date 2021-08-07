"""Multinomial Naive Bayes (MNB)"""

from skopt.space import Real
from sklearn.naive_bayes import MultinomialNB

def get_instance():
    return MultinomialNB()

def get_name():
    return "Multinomial Naive Bayes"

def get_initials():
    return "MNB"

def get_search_spaces():
    return {
        'alpha': Real(1e-3, 1e3),
        'fit_prior': [True, False]
    }