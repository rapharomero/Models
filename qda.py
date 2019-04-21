import numpy as np


class QDA(object):
    """
    Class for quadratic discriminant analysis
    """
    def __init__(self):
        self.pi = None
        self.m0 = None
        self.m1 = None
        self.sigma0 = None
        self.sigma1 = None

    def fit(self, X_tr, y_tr):
        """
        Trains the quadratic model using MLE
        """

        # N samples
        n = np.shape(X_tr)[0]
        n1 = X_tr[y_tr == 1].shape[0]
        n0 = X_tr[y_tr == 0].shape[0]
        # MLE Means
        m0 = np.mean(X_tr[y_tr == 0], axis=0)
        m1 = np.mean(X_tr[y_tr == 1], axis=0)

        # MLE class 1 probability
        self.pi = float(np.shape(X_tr[y_tr == 1])[0]) / n

        # MLE Covariance
        sigma1 = np.dot((X_tr[y_tr == 1] - m1).T, X_tr[y_tr == 1] - m1) / n1
        sigma0 = np.dot((X_tr[y_tr == 0] - m0).T, X_tr[y_tr == 0] - m0) / n0

        # Save parameters
        self.m0 = m0
        self.m1 = m1
        self.sigma0 = sigma0
        self.sigma1 = sigma1

        return m0, m1, sigma0, sigma1

    def get_params(self):
        """
        Returns the model parameters
        """
        return self.m0, self.m1, self.sigma0, self.sigma1, self.pi

    def get_coefficients(self):
        """
        Computes the quadratic coeficients C, w, b based on the model parameters
        C, w, b are used to compute the estimated conditional probability of y given x in the QDA case
        """
        det0 = np.linalg.det(self.sigma0)
        det1 = np.linalg.det(self.sigma1)
        A0 = np.linalg.inv(self.sigma0)
        A1 = np.linalg.inv(self.sigma1)
        m0 = self.m0
        m1 = self.m1

        b = (np.dot(m0,np.dot(A0, m0)) - np.dot(m1, np.dot(A1, m1))) / 2 - math.log((1 - pi) / pi) - math.log(det1 / det0)/2
        w = np.dot(A1, m1) - np.dot(A0, m0)
        C = (A0 - A1)/2
        return C, w, b

    def predict_proba(self, X_t):
        """
        Predicts the output class probabilities given the input X_t
        """
        C, w, b = self.get_coefficients()
        return(sigmoid(quadratic_function(X_t, C, w, b)))

    def predict(self, X_t, treshold = 0.5):
        """
        Predicts the output class given the input X_t
        """
        y_prob = self.predict_proba(X_t)
        return (y_prob > treshold).astype(int)
