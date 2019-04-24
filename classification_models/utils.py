import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib import cm
import pylab as pl
from math_functions import *

# Data loading function
def extractdata(classification, testfile):
    """ extrac the trainning data end test data given the path
    to these files"""
    train  = pd.read_table(classification, header=None, engine='python')
    test = pd.read_table(testfile, header=None, engine='python')
    X_train, X_test = train.values[:,0:2], test.values[:,0:2]
    y_train, y_test = train.values[:,2], test.values[:,2]
    return X_train, y_train, X_test, y_test

## Plot functions

def plot_data(data, labels=None, markers = ['o', 's']):
    """
    Plot 2d points representing the data
    """
    if labels is None:
        plt.scatter(data[:, 0], data[:, 1], c='b', s = 80, marker = markers[0])

    else:
        classes = np.sort(np.unique(labels))
        n_classes = classes.shape[0]
        color_blind_list = sns.color_palette("colorblind", n_classes)
        sns.set_palette(color_blind_list)

        for i, l in enumerate(classes):
            plt.scatter(data[labels == l, 0],
                        data[labels == l, 1],
                        c=color_blind_list[i],
                        s=80,
                        marker=markers[i])

def make_grid(data=None, xmin=-5, xmax=5, ymin=-5, ymax=5, n_points = 400):
    """
    Computes a plotting grid given the data.
    xmin, xmax, ymin, ymax is used by default if no data is given as input
    """
    if data is not None:
        xmin, ymin = np.min(data, axis = 0)
        xmax, ymax = np.max(data, axis = 0)

    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)

    x, y = np.meshgrid(np.linspace(xmin, xmax, n_points), np.linspace(ymin, ymax, n_points))
    grid = np.c_[x.ravel(), y.ravel()] # grid has n_points ^2 row and 2 columns
    return x, y, grid

def plot_separator(data, w, b, C=None):
    """
    Plots the decision separator
    """
    x, y, grid = make_grid(data)
    if(C is None):
        C = np.zeros((w.shape[0], w.shape[0]))
    f = lambda x: quadratic_function(x, C, w, b)

    color_blind_list = sns.color_palette("colorblind", 2)
    sns.set_palette(color_blind_list)

    plt.contour(x, y, f(grid).reshape(x.shape),  levels = 0, alpha = 1)
    plt.show()
