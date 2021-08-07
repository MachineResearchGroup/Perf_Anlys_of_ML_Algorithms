import nltk as nlp
import pandas as pd
from tools import data
from nltk.stem import WordNetLemmatizer
from settings.system_settings import has_processed_data


def run():
    if has_processed_data():
        user_say = input("\nYou want to run a new natural language processing? [yes/not]\n")
        if user_say == "yes":
            __run__()
    else:
        __run__()


def __run__():
    print("\nRun natural language processing...")
    corpus = data.get_dataset()
    requirements = corpus['RequirementText']
    new_corpus = pd.DataFrame()
    for i, string in enumerate(requirements):
        if string != '':
            str_help = string.lower()
            str_help = tokenization(str_help)
            str_help = normalization(str_help)
            str_help = lemmatization(str_help)
            new_corpus.loc[i, 'RequirementText'] = str(str_help)

    new_corpus['RequirementText'] = new_corpus['RequirementText'].str.replace('[^\w\s]', "", regex=True)
    new_corpus['Class'] = corpus['Class']
    new_corpus = new_corpus.sample(frac=1).query("Class != 'F'").dropna()
    data.export_to_csv("./datasets/", "PROMISE_exp_preprocessed.csv", new_corpus)


def tokenization(text):
    return nlp.word_tokenize(text)


def normalization(text):
    stop_words = nlp.corpus.stopwords.words('english')
    return [word for word in text if word not in stop_words and word.isalpha()]


def lemmatization(text):
    lemma = WordNetLemmatizer()
    return [lemma.lemmatize(word) for word in text]

