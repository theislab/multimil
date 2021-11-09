import numpy as np

def get_split_idx(class_label):
    labels = set(list(class_label))

    idx = []
    for label in labels:
        idx.append(np.where(class_label == label)[0][-1]+1)

    idx.sort()
    idx = idx[:-1]

    return idx
