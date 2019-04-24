import numpy as np
from math_functions import *
#Logistic regression

class LogisticRegression(object):
    """
    Class for logistic regression
    """

    def __init__(self):
        self.w = None

    def gradient(self, X_tr, y_tr, w):
        """
        Computes the gradient and the hessian of the log likelihood of (X_t, y_t) with respect to parameter w
        """
        n = np.shape(X_tr)[0]

        Xe = np.c_[X_tr, np.ones(n)] # Append X_tr with ones to include bias term

        eta = sigmoid(np.dot(Xe, w.T))

        grad = np.dot(Xe.T, np.reshape((y_tr - eta), (n, 1)))

        sigma = np.diag(eta * (1 - eta))

        hessian = np.dot(- Xe.T, np.dot(sigma, Xe))

        return grad, hessian


    def fit(self, X_tr, y_tr, tol = 0.001):
        """
        Trains the logistic regression model using the IRLS (Newton- Raphson) algorithm
        """
        w = np.zeros((X_tr.shape[1] + 1)) # bias included in the coefficient vector

        g, H = self.gradient(X_tr,y_tr, w)

        while (np.linalg.norm(g) > tol):
            # Newton-Ralphson step
            w = w - np.ravel(np.dot(np.linalg.inv(H), g))

            # Update gradient and Hessian
            g, H = self.gradient(X_tr, y_tr, w)

        self.w = w
        return w

    def predict_proba(self, X_t):
        """
        Predicts the output class probabilities of the input X_t
        """
        return sigmoid(np.dot(X_t, (self.w[:2]).T) + self.w[2])

    def predict(self, X_t, treshold = 0.5):
        """
        Predicts the output class of the input X_t
        """
        y_prob = self.predict_proba(X_t)
        return (y_prob > treshold).astype(int)
