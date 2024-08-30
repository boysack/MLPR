import numpy as np
from numpy import ndarray
import os
from modules.utils.operations import col

def load(filename: str, delimiter: str = ",") -> tuple[ndarray, ndarray, dict]:
    # TODO: check if it's convenient to return even an inversed version of label_dictionary
    """
    Load the dataset from a file.

    Parameters:
    filename (str): filename from which extract the data.
    delimiter (str): delimiter used in the file to separate the data values.

    Returns:
    ndarray: the dataset (sample = column vector).
    ndarray: the labels.
    dict: the labels dictionary.
    """
    if not os.path.isfile(filename):
        raise Exception("Must insert a valid filename")
    # used to create label dictionary
    label_index = 0
    label_dict = {}
    with open(filename, "r") as f:
        first = True
        while True:
            line = f.readline()
            # if EOF, break
            if not line:
                break
            line = line.split(delimiter)

            # if first iteration, then instantiate the numpy arrays
            if first:
                D = np.empty((len(line)-1, 0), dtype=float)
                L = np.empty((0), dtype=int)
                first = False

            # create the label entry if not exists
            label = line[-1].strip()
            if label not in label_dict.keys():
                label_dict[label] = label_index
                label_index += 1

            # append the sample in the arrays
            D = np.hstack((D, col(np.array([float(i) for i in line[:-1]]))))
            L = np.append(L, label_dict[label])

    # if binary problem with class 0 and 1, transform in True and False their labels name
    if all([True if key=="0" or key=="1" else False for key in label_dict.keys()]):
        if label_dict["1"] == 0:
            # must invert the labels
            L = np.array([1 if l == 0 else 0 for l in L])

        label_dict = {}
        label_dict["False"] = 0
        label_dict["True"] = 1

    return D, L, label_dict

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]

    return (DTR, LTR), (DVAL, LVAL)
