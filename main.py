from tools import plot
from tools import time
from tools import compute
from embedding import tfidf
from embedding import vectorizer
from nlp import text_preprocessing
from settings import system_settings
from classification_task import training

k_folds = 5
n_inter = 10

if __name__ == "__main__":
    total_time = time
    total_time. init()
    print("\nStart at "  +total_time.get_init_date_time())
    system_settings.run()
    text_preprocessing.run()
    vectorizer.run(tfidf)
    for i in range(n_inter):
        print("\nInteraction "+str(i+1))
        training.run(k_folds, i+1)
    total_time.end()
    compute.general_mean(n_inter, k_folds)
    print("\nFinished at "  +total_time.get_end_date_time() + 
    '\nTotal run time: ' + total_time.get_execu_time())
    # plot.box("classification_results.csv")