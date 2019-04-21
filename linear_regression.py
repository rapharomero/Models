import numpy as np



#linear regression
class LinearRegression(object):
    """
    Class for linear regression
    """

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X_tr, y_tr):
        """
        Trains the linear model using MLSE
        """
        n,p = np.shape(X_tr)
        ones = np.ones((n, 1))

        X_aug = np.c_[X_tr, ones]

        M = np.dot(X_aug.T, X_aug)

        b = np.dot(X_aug.T, y_tr)

        w_aug = np.linalg.solve(M, b)

        self.w = w_aug[0:p]
        self.b = w_aug[p]
        return self.w, self.b

    def get_params(self):
        return self.w, self.b

    def predict_regressor(self, X_t):
        """
        Predicts the ouput of the linear function given the input X_t
        """
        return X_t.dot((self.w).T) + self.b
    def predict(self, X_t, treshold = 0.5):
        """
        Predicts the binary class of X based on treshold
        """
        return (X_t.dot((self.w).T) + self.b > treshold).astype(int)
