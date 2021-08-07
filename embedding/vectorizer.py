from sys import path
from tools import data
from settings.system_settings import has_vectorized_data

def run(approach):
    """approach -> TF-IDF or Word2Vec"""

    if all(has_vectorized_data()):
        user_say = input("\nYou want to run a new embedding? [yes/not]\n")
        if user_say == "yes":
            __run__(approach)
    else:
        __run__(approach)


def __run__(approach):
    path = "./datasets/"
    print('\nVectoring the requirements text...')
    x, y = approach.vectorize()
    data.export_to_npz(path, "tfidf_requirements", x)
    data.export_to_npy(path, "tfidf_classes", y)