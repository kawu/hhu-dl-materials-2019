from typing import Dict, List, Tuple, Iterable


import os
import os.path
import random
import csv


def read_lines(path: str) -> List[str]:
    """Return the list of lines in the file under the given path."""
    lines = []
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


# Some useful type aliases: languages are represented by strings,
# as well as person names.
Lang = str
Name = str


# Two different ways to represent the dataset:
#
# (a) dictionary mapping languages to the corresponding names
#     (close to the actual representation on disk)
#
# (b) list of pairs (name, language), which is the representation
#     we can actually use for training a machine learning model
#
DataDict = Dict[Lang, List[Name]]
DataList = List[Tuple[Name, Lang]]


def read_names(dir_path: str) -> DataDict:
    """
    Read the dataset in the given directory.  The result is the
    dictionary mapping languages to the corresponding person names.
    """
    data = {}
    for file_path in os.listdir(dir_path):
        full_path = os.path.join(dir_path, file_path)
        lang = os.path.splitext(file_path)[0]
        data[lang] = []
        for name in read_lines(full_path):
            data[lang].append(name)
    return data


# A toy data dictionary to test our functions
data_dict: DataDict = {
    "EN": ["Andrew", "Burford", "Downey", "Kilford", "Travis"],
    "DE": ["Adlersflügel", "Brahms", "Günther", "Krüger", "Schulte"]
}


# TODO: it's your job to implement this function.
def convert(data: DataDict) -> DataList:
    """
    Convert the dictionary representation to the list
    of (name, language) pairs.
    """
    result = []
    for (lang, name_list) in data.items():
        for name in name_list:
            result.append((name, lang))
    return result


# Some checks to see if conversion works as expected
data_list = convert(data_dict)
assert ('Travis', 'EN') in data_list
assert ('Schulte', 'DE') in data_list
assert ('Adlersflügel', 'EN') not in data_list


# TODO: it's your job to implement this function.
def random_split(data: list, rel_size: float) -> Tuple[list, list]:
    """
    Take a list of elements (e.g., a DataList) and divide it randomly to
    two parts, where the size of the first part should be roughly equal
    to `rel_size * len(data)`.

    Arguments:
    data: list of input elements
    rel_size: the target relative size of the first part
    """
    # Check the input argument
    assert rel_size >= 0 and rel_size <= 1
    # We don't want to modify the input list, hence we create a copy
    copy = data[:]
    random.shuffle(copy)
    # Target size of the first part
    k = round(rel_size * len(data))
    return (copy[:k], copy[k:])


# Check if random_split works as expected on the toy dataset.
(part1, part2) = random_split(data_list, rel_size=0.5)
# The length of the two parts should be the same (since rel_size=0.5)
assert len(part1) == len(part2)
# Since there are no repetitions, there should be no common elements
# in the two parts
assert not set(part1).intersection(set(part2))


# TODO: it's your job to implement this function.  Hint: you can do that
# with the help of the random_split function.
def three_way_split(data: list, dev_size: float, test_size: float) \
        -> Tuple[list, list, list]:
    """
    Take a list of elements (e.g., a DataList) and divide it randomly to
    three parts (train, dev, test), where the size of the dev part should
    be roughly equal to `dev_size * len(data)` and the size of the test part
    should be roughly equal to `test_size * len(data)`
    """
    assert dev_size >= 0 and test_size >= 0
    assert dev_size + test_size <= 1
    (dev_test, train) = random_split(data, dev_size + test_size)
    (dev, test) = random_split(dev_test, dev_size / (dev_size + test_size))
    return (train, dev, test)


# TODO: it's your job to implement this function.  You can choose an
# appropiate format (what is important is that the dataset is easy
# to load later).  Hint: you can for instance use the csv module
# (https://docs.python.org/3/library/csv.html).
def save_data(data: Iterable[list], file_path: str) -> None:
    """Save the give dataset in the given file."""
    with open(file_path, 'w') as file:
        csv_writer = csv.writer(file, delimiter=',')
        for elem in data:
            csv_writer.writerow(elem)


# TODO: combine the implemented functions to actually divide the dataset
# with names to three separate parts.  You can use 80% of the original
# dataset as train and 10% of the original dataset as dev.
all_data = convert(read_names("all_data"))
(train, dev, test) = three_way_split(all_data, dev_size=0.1, test_size=0.1)
save_data(all_data, "split/all.csv")
save_data(train, "split/train.csv")
save_data(dev, "split/dev.csv")
save_data(test, "split/test.csv")
