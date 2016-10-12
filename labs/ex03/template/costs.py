# -*- coding: utf-8 -*-
"""a function used to compute the cost."""

import numpy as np

def compute_cost_mae(y, tx, w):
    """calculate the cost.
    you can calculate the cost by mae.
    """
    e = np.absolute(y - np.dot(tx,w)) 
    toReturn = (np.sum(e) / y.shape[0])
    return toReturn

def compute_cost_mse(y, tx, w):
    """calculate the cost.
    you can calculate the cost by mse.
    """
    e = y - np.dot(tx,w) 
    toReturn = (np.dot(e.T, e) / (2*y.shape[0]))
    return toReturn