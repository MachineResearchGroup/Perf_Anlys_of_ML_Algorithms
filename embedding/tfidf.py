from tools import data
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize():
    file_name = "PROMISE_exp_preprocessed.csv"
    classes = data.get_classes(file_name)
    requirements = data.get_requirements(file_name)
    vetorClass = LabelEncoder()
    y_class = vetorClass.fit_transform(classes)
    vetorText = TfidfVectorizer()
    x_tokens = vetorText.fit_transform(requirements, classes)
    return x_tokens, y_class
