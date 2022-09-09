import numpy as np


def flip_binary_class(data):
    data.class_names = np.flip(data.class_names)
    data.data[data.target_name] = 1 - data.data[data.target_name]
    return data