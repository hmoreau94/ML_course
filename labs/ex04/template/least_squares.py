# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
from costs import *


def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # returns mse, and optimal weights
    # ***************************************************
    transpose = tx.T
    w = np.linalg.solve(np.dot(transpose,tx),np.dot(transpose,y))
    #compute mse
    mse = compute_mse(y,tx,w)
    return (mse,w)
