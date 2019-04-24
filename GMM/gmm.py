# Definition of the gaussian density function with non isotropic covariance matrix
def gaussian(X, mu, sigma):
    """
    Computes the images of the rows of X by the gaussian of mean mu and covariance S

    Parameters
    ----------
    X : input matrix, of dimension (n, p). Each row represents an input
    mu : mean of the gaussian, of dimension p
    S : covariance of the gaussian
    Returns
    -------
    Array of size n containing the values
    """
    p = X.shape[1]
    Sinv = np.linalg.pinv(sigma)
    det = np.linalg.det(sigma)
    const = (1.0 / np.sqrt((2 * np.pi * det) ** p))
    return const * np.exp(-0.5 * np.einsum("ij,ij->i", (X - mu), Sinv.dot((X - mu).T).T))

def random_select(X, k):
    """
    Selects k random points among the observations
    """
    idx = np.random.randint(np.shape(X)[0], size = k)
    return X[idx]


def random_uniform(X, k):
    """
    Returns k points picked uniformly at random on the smallest rectangle containing all the observations
    """
    x0 = np.random.uniform(np.amin(X[:,0]),np.amax(X[:,0]) , k)
    y0 = np.random.uniform(np.amin(X[:,1]),np.amax(X[:,1]) , k)
    C0 = np.c_[x0, y0]
    return C0


def tauNonIso(i,j,pi,mu,sigma):
    """
    Computes the weight
    Parameters
    ----------
    i : index of observation
    j : index of  class
    pi : distribution of z
    mu : means of the distributions of x given z
    sigma : covariance matrices of the distributions of x given z

    Returns
    -------
    Value of the weight of th
    """
    K = np.shape(pi)[0] # Number of classes
    num = pi[j] * gaussNonIso(X[i], mu[j], sigma[j])
    denom = np.sum([pi[j]*gaussNonIso(X[i],mu[j],sigma[j]) for j in range(K)])
    return num/denom


class GMM(object):
    """
    Class for Gaussian mixture model.
    """

    def __int__(self):
        self.n_classes = K
        self.means = None
        self.sigmas = None
        self.weights = None
        self.log_likelihoods = None

    def log_likelihood(X, tau, pi, means, sigmas):
        """
        Log-likelihood of the GMM
        """
        p = np.shape(X)[1]
        n, K = np.shape(tau)
        res = np.sum(np.sum(tau))

        inverses = np.linalg.pinv(sigmas)
        dets = np.linalg.det(sigmas)

        const = np.log(2 * np.pi * dets) ** (p / 2)
        terms = tau * np.log(pi) - const - np.einsum('ik,kji->ik',
                                                      np.einsum('ik,kip -> ik',
                                                                tau,
                                                                np.stack(X-mu for mu in means)),
                                                      np.stack(np.dot(sinv, (X - mean).T) for mean, sinv in zip(means, inverses)))

        return np.sum(np.sum(terms))

    def E_step(self, X, pi, means, sigmas):
        """
        Performs the E step on observations X using the current set of parameters
        """
        # Store a n * k matrix containing the image of each observation i by the j th gaussian, weighted by pi
        gaussian_values = np.vstack([pi[j] * gaussian(X, means[j], sigmas[j]) for j in range(pi.shape[0])]).T

        # Normalize these values with the sum along the second axis
        tau = np.einsum('i,ik->ik', 1.0 / np.sum(gaussian_values, axis = 1), gaussian_values)
        return tau

    def M_step(self, X, tau):
        """
        Performs the M step
        """
        pi = np.mean(tau, axis = 0)
        means = np.einsum('k,kp->kp', np.sum(tau, axis=0), np.dot(tau.T, X))
        sigmas = np.einsum('k,klm->klm',
                           1.0 / np.sum(tau, axis=0),
                           np.stack([np.dot(np.einsum('i,ik->ik', tau[:, j], X - mu).T, X - mu) for j,mu in enumerate(means)]))

        return pi, means, sigmas

    def fit(self, X, K=5, initialize_f=random_select, Niter=100):
        """
        Trains the GMM sing the Expectation-Maximization algorithm.

        Parameters
        ----------
            X : Observations
            K : number of classes
            Niter : number of iterations

        Returns
        -------
            mu : Array containing the means of each class
            sigma : Array containing the standard deviations of each class
        """
        n, p = X.shape
        self.n_classes=K

        if initialize_f is None:
        # Default: Random initialization
            pi = np.random.rand(K)
            pi /= np.sum(pi)
            means = X[np.random.randint(n, size=K)] # Pick K data points at random to initialize the means of the gaussians
        else:
            # Custom initialization
            pi = np.random.rand(K)
            pi /= np.sum(pi)
            means = initialize_f(X, K)

        # Initialize covariances
        sigmas = np.stack([np.identity(p) * (i + 1) for i in range(K)], axis = 0)


        # EM Loop
        log_likelihoods = []
        labels = range(K)
        for t in range(Niter):
            # E-step
            tau = self.E_step(X, pi, means, sigmas)
            # M-step
            pi, means, sigmas = self.M_step(X, tau)
            # Log Likelihood
            #log_likelihoods += self.log_likelihood(X, tau, pi, means, sigmas)

        self.pi = pi
        self.means = means
        self.sigmas = sigmas
        self.log_likelihoods = log_likelihoods

        return means, sigmas

    def predict(self, X_t):
        """
        Predicts the most likely class for each element in X_t
        """
        gaussian_values = np.vstack([gaussian(X_t, self.means, self.sigmas) for j in range(self.n_classes)])
        return np.argmin(gaussian_values, axis=1)
