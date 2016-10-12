# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Stochastic Gradient Descent
"""
from helpers import *
from gradient_descent import *


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    #We simply compute the normal gradient of this batch
    
    #Is this what we should do?
    #toReturn = np.sum([compute_gradient(yn,txn,w) for (yn,txn) in zip(y,tx)])/len(y)
    return compute_gradient(y,tx,w)

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_epochs, gamma):
    """Stochastic gradient descent algorithm."""
    batches = batch_iter(y, tx, batch_size)
    
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_epochs):
        try:
            # get the next item
            y,tx = batches.__next__()
            
        except StopIteration:
            # No more elements in Iterator, we get a new one
            batches = batch_iter(y, tx, batch_size)
            y,tx = batches.__next__()
            
        gradient = compute_stoch_gradient(y,tx,w)
        loss = compute_cost(y,tx,w)
        w_new = w - gamma*gradient
        w = w_new
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_epochs - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws
