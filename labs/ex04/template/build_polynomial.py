# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # returns an array of dimension N*degree
    # ***************************************************
    rx = x.reshape((-1,1)) #we want the data to be organised in line (first line xi^0 xi^1 xi^2 ...xi^degree)
    toReturn = np.ones((rx.shape[0],1))
    for i in range(degree):
        toReturn = np.hstack([toReturn,np.power(rx,i+1)])
    return toReturn

