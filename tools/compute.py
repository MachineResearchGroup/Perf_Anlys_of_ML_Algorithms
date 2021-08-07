import pandas as pd
from tools import data
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from settings.system_settings import has_file
import classifiers


def hyper_results(model, model_name, fold, time):
    print("\nComputing hyper results...")
    path = "./results/csv/"
    file_name = "hyper_results.csv"
    results = {
        'Algorithm': model_name,
        'f1-score Macro': '{:.0%}'.format(model.best_score_),
        'Fold': str(fold),
        'Splits': model.n_splits_,
        'Interactions': model.n_iter,
        'Execution time': time.get_execu_time(),
        'Initial Date/Hour': time.get_init_date_time(),
        'Final Date/Hour': time.get_end_date_time(),
        'Params': model.best_params_
    }
    __write_data(path, file_name, results)
    



def classification_results(y_test, predict, classifier, time, fold, inter):
    print("\nComputing classification results...")
    accuracy = accuracy_score(y_test, predict)
    precision = precision_score(y_test, predict, average='macro', zero_division=0)
    recall = recall_score(y_test, predict, average='macro', zero_division=0)
    f1 = f1_score(y_test, predict, average='macro', zero_division=0)
    path = "./results/csv/"
    file_name = "classification_results.csv"
    results = {
        'Algorithm': classifier.get_name(),
        'Initials': classifier.get_initials(),
        'Interaction': str(inter),
        'Fold': str(fold),
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Execution time': time.get_execu_time(),
        'Initial Date/Hour': time.get_init_date_time(),
        'Final Date/Hour': time.get_end_date_time()
    }

    __write_data(path, file_name, results)


def mean_per_interaction(inter, k_folds):
    print("\nComputing mean per interaction...")
    path = "./results/csv/"
    file_name = "classification_results.csv"
    classification_results = data.get_results(file_name)
    classifiers = classification_results["Algorithm"].unique()
    for clf in classifiers:
        helper = classification_results.query("Interaction == "+str(inter)+"and Algorithm == '"+clf+"'")
        results = {
            'Algorithm': clf,
            'Interaction': str(inter),
            'k_folds': str(k_folds),
            'Accuracy': helper['Accuracy'].mean(),
            'Precision': helper['Precision'].mean(),
            'Recall': helper['Recall'].mean(),
            'F1': helper['F1'].mean()
        }

        file_name = "mean_per_interaction.csv"
        __write_data(path, file_name, results)


def general_mean(n_inter, k_folds):
    print("\nComputing general mean for "+ str(n_inter) +" interaction...")
    path = "./results/csv/"
    file_name = "classification_results.csv"
    classification_results = data.get_results(file_name)
    classifiers = classification_results["Algorithm"].unique()
    for clf in classifiers:
        helper = classification_results.query("Algorithm == '"+clf+"'")
        results = {
            'Algorithm': clf,
            'Interactions': str(n_inter),
            'k_folds': str(k_folds),
            'Accuracy': helper['Accuracy'].mean(),
            'Precision': helper['Precision'].mean(),
            'Recall': helper['Recall'].mean(),
            'F1': helper['F1'].mean()
        }

        file_name = "general_mean.csv"
        __write_data(path, file_name, results)


def __write_data(path, file_name, results):
    if has_file(path, file_name):
        data.write_row(path, file_name, results.values())
    else:
        data_frame = pd.DataFrame(columns=results.keys())
        data_frame = data_frame.append(results, ignore_index=True)
        data.export_to_csv(path, file_name, data_frame)