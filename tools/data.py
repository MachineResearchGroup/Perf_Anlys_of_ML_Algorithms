import csv
import numpy
import pandas as pd
from scipy import sparse

__base_path = './datasets/'
__dataset_path = './datasets/PROMISE_exp.csv'

def get_dataset():
    return pd.read_csv(__dataset_path, encoding='utf-8')


def get_requirements(file_name):
    return pd.read_csv(__base_path+file_name, encoding='utf-8')['RequirementText']


def get_classes(file_name):
    return pd.read_csv(__base_path+file_name, encoding='utf-8')['Class']


def get_encoded_requirements(file_name):
    return sparse.load_npz(__base_path+file_name)


def get_encoded_classes(file_name):
    return numpy.load(__base_path+file_name)

def get_results(file_name):
    return pd.read_csv("./results/csv/"+file_name, encoding='utf-8')

def write_row(path, file_name, data):
    with open(path+file_name, 'a') as archive:
        writer = csv.writer(archive)
        writer.writerow(data)

def export_to_csv(path, file_name, data_frame):
    data_frame.to_csv(path+file_name, index=False)


def export_to_npy(path, file_name, data):
    numpy.save(path+file_name, data)        


def export_to_npz(path, file_name, data):
    sparse.save_npz(path+file_name, data)