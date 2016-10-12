# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
#from build_polynomial import *

def build_poly(x, degree):
    """polynomial basis function."""
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

def plot_fitted_curve(y, x, beta, degree, ax):
    """plot the fitted curve."""
    ax.scatter(x, y, color='b', s=12, facecolors='none', edgecolors='r')
    xvals = np.arange(min(x) - 0.1, max(x) + 0.1, 0.1)
    tx = build_poly(xvals, degree)
    #tx = np.c_[np.ones((len(xvals), 1)), build_poly(xvals, degree)]
    f = tx.dot(beta)
    ax.plot(xvals, f)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Polynomial degree " + str(degree))
