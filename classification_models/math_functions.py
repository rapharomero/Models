import numpy as np


def quadratic_form(x, A):
    """
    Computes the image of x by the quadratic form represented by a symetric matrix A
    """
    if(len(x.shape) == 1):
        x = x[np.newaxis, ...]
    return np.einsum('ij,ij->i', x, np.dot(x, A))

def quadratic_function(x, A, w, b):
    """
    Computes the image of x by the quadratic function represented by the symetric matrix A,
    the vector w, and the bias b
    """
    return(quadratic_form(x, A) + np.dot(x, w) + b)

def linear_function(x, w, b):
    """
    Computes the image of x by the linear function represented by coefficients w and b
    """
    return x.dot(w.T) + b

def sigmoid(x):
    """
    Sigmoid function
    """
    return 1/(1 + np.exp(-x))

def decision(y):
    return (np.array(y) > 0.5).astype(int)
