# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    index_array = np.arange(len(x))
    np.random.shuffle(index_array) #shuffles the indices
    bound = np.floor(len(index_array)*ratio)
    #print("Training : [0,",bound-1,"] \t Testing : [",bound,len(y),"]")
    x_training = x[index_array[:bound]]
    y_training = y[index_array[:bound]]
    x_testing = x[index_array[bound:]]
    y_testing = y[index_array[bound:]]
    training = (x_training,y_training)
    testing = (x_testing,y_testing)
    return (training,testing)
    
