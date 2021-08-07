import os
import nltk as nlp

__home = os.environ['HOME']

def run():
    print("\nConfiguring the dependencies...")
    nltk_downloads()
    create_directories()


def nltk_downloads():
    path = __home+'/nltk_data'
    if not os.path.isdir(path):
        nlp.download('punkt')
        nlp.download('stopwords')
        nlp.download('wordnet')


def create_directories():
    paths = [
        "./results", 
        "./results/csv", 
        "./results/images"
        ]
    for path in paths:
        if not os.path.isdir(path):
            os.mkdir(path)

def has_processed_data():
    return os.path.isfile("./datasets/PROMISE_exp_preprocessed.csv")


def has_file(path, file_name):
    return os.path.isfile(path+file_name)


def has_vectorized_data():
    """return [true, true] if there is the requirements.npz and classes.npy files"""

    return os.path.isfile("./datasets/tfidf_requirements.npz"), \
    os.path.isfile("./datasets/tfidf_classes.npy")
