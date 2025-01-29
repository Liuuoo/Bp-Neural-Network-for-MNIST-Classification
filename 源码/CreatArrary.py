import numpy as np

def create_edge(size1, size2):
    array = np.random.uniform(0.005, 0.01, size=(size1, size2))
    array = np.round(array, 4)
    return array

def create_node(size):
    array = np.zeros((size, 1))
    array = np.round(array, 4)
    return array