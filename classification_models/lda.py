import numpy as np
from math_functions import *

#LDA FUNCTIONS
class LDA(object):
    """
    class for binary Linear Discriminant Analysis
    """

    def __init__(self):
        self.m0 = None
        self.m1 = None
        self.sigma = None
        self.pi = None

    def fit(self, X_tr, y_tr):
        """
        Takes the trainning data, estimates sigma, m0, m1 and returns the coefficients
        w, b and pi
        """
        # N samples
        n = np.shape(X_tr)[0]

        # MLE Means
        self.m0 = np.mean(X_tr[y_tr == 0], axis=0)
        self.m1 = np.mean(X_tr[y_tr == 1], axis=0)

        # MLE class 1 probability
        self.pi = float(np.shape(X_tr[y_tr == 1])[0]) / n

        # MLE Covariance
        sigma1 = np.dot((X_tr[y_tr == 1] - self.m1).T, X_tr[y_tr == 1] - self.m1)
        sigma0 = np.dot((X_tr[y_tr == 0] - self.m0).T, X_tr[y_tr == 0] - self.m0)

        self.sigma = (sigma1 + sigma0) / n

        return self.m0, self.m1, self.sigma, self.pi

    def get_separator(self):
        """
        Returns the separator coefficients w, b
        """
        sigma_inv = np.linalg.inv(self.sigma)
        w = np.dot(sigma_inv, self.m1 - self.m0)
        b = 0.5 * (quadratic_form(self.m0, sigma_inv) - quadratic_form(self.m1, sigma_inv)) - np.log((1 - self.pi) / self.pi)
        return w, b

    def get_params(self):
        """
        Returns the current parameters of the model
        """
        return self.m0, self.m1, self.sigma, self.pi

    def predict_proba(self, X_t):
        """
        Predicts the probability of the class "1" for each line of X_test
        """
        w, b = self.get_separator()
        arg = np.dot(X_t, w) + b
        y_prob = sigmoid(np.dot(X_t, w) + b)
        return y_prob

    def predict(self, X_t, probability_treshold = 0.5):
        """
        Predicts the classes of the elements in X_test given a probability treshold
        """
        y_prob = self.predict_proba(X_t)
        return (y_prob > probability_treshold).astype(int)
