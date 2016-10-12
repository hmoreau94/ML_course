# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    transpose = tx.T
    lambdaIden = lamb*np.eye(tx.shape[1])
    LHS = np.dot(transpose,tx)+lambdaIden
    RHS = np.dot(transpose,y)
    beta = np.linalg.solve(LHS,RHS)
    return beta
