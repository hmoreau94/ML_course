# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Gradient Descent
"""
import numpy as np
import costs

def compute_gradient(y,tx,w):
    """Compute the gradient with MAE."""
    e = y - np.dot(tx,w.T)
    s = np.sign(e).reshape(-1,1)
    toReturn = np.sum(-tx*s,axis=0)/len(y) 
    return toReturn
            

def compute_gradient1(y, tx, w):
    """Compute the gradient with MSE."""
    e = y - np.dot(tx,w.T)
    toReturn = -np.dot(tx.T,e) / y.shape[0]
    return toReturn
#to test
#compute_gradient(np.array([1,3]),np.array([[1,4],[1,5]]),np.array([1,2]))


def gradient_descent(y, tx, initial_w, max_iters, gamma): 
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y,tx,w)
        loss = compute_cost(y,tx,w)
        w_new = w - gamma*gradient
        w = w_new
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
