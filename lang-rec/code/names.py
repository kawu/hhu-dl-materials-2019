import csv

from core import DataSet


def load_data(file_path: str) -> DataSet:
    """Load the dataset from a .csv file."""
    data_set = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for pair in csv_reader:
            data_set.append(pair)
    return data_set
