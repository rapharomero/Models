import numpy as np


# gaussian density
def gauss(x,mu,S):
    rv = multivariate_normal(mu,S)
    return rv.pdf(x)

# gaussian density in log scaleù^m
def gauss_log(x,mu,S):
    rv = multivariate_normal(mu,S)
    return rv.logpdf(x)


class HMM(object):
    """
    Class for Hidden Markov model
    """
    def __init__(self):
        self.pi1 = None # Initial class probabilities
        self.A = None # Transition matrix
        self.means = None # Means of the gaussians
        self.sigmas = None # Covariances of the gaussians

    def alpha_log(self, U, pi1, A, means, sigmas):
        """
        Computes the alpha messages recursively
        """
        T = np.shape(U)[0] # Number of observations
        K = np.shape(pi1)[0] # Number of classes
        A_log = np.log(A) # Transition matrix in logscale

        # Initialization
        logalphas = np.zeros((T, K))
        logalphas[0] = np.log(pi1) + np.hstack([gauss_log(U[0], mu, sigma) for (mu, sigma) in zip(means, sigmas)])

        log_emissions = np.vstack([gauss_log(U, mu, sigma) for (mu, sigma) in zip(means, sigmas)]).T
        # Alpha recursions
        for t in range(1, T):
            for i in range(K):
                logalphas[t][i] = log_emissions[t, i] + logsumexp(A_log[i] + logalphas[t-1])

        return logalphas


    def beta_log(self, U, pi1, A, means, sigmas):
        """
        Computes the beta messages recursively
        """
        T = np.shape(U)[0] # Number of observations
        K = np.shape(pi1)[0] # Number of classes
        A_log = np.log(A) # Transition matrix in logscale
        # Initialization
        logbetas = np.zeros((T, K))
        logbetas[T-1] = np.zeros(np.shape(pi1))

        # Beta recursions
        for t in range(1, T):
            for j in range(K):
                log_ems = np.array([gauss_log(U[T-t], mu, sigma) for (mu, sigma) in zip(means, sigmas)])
                logbetas[T-t-1][j] = logsumexp(A_log[:, j] + log_ems + logbetas[T-t])

        return logbetas


    def E_step(self, U, log_alphas, log_betas):
        """
        Performs the E step of the EM for HMM
        """
        T,d = np.shape(U)
        K = np.shape(self.A)[1]
        A_log = np.log(self.A)

        # Log-Likelihood of the observations
        likelihoods = logsumexp(log_alphas + log_betas, axis=1)

        # Compute marginal posterior probabilities in log scale

        marg_post = log_alphas + log_betas - likelihoods[..., np.newaxis]

        # Compute joint posterior probabilities
        gauss_logs = np.stack([gauss_log(U[1:], mu, sigma) for (mu, sigma) in zip(self.means, self.sigmas)])

        joint_messages = log_alphas[:(T-1), :, np.newaxis].swapaxes(1,2) + log_betas[1:, :, np.newaxis]

        joint_post = joint_messages + A_log  + gauss_logs[...,np.newaxis].swapaxes(0,1) - likelihoods[1:, np.newaxis, np.newaxis]

        return np.exp(marg_post), np.exp(joint_post)

    def M_step(self, U, tau, nu):
        """
        Performs the M-step of the EM on the Hidden markov model
        """
        T, K = np.shape(tau)
        pi1 = tau[1]

        self.A = np.sum(nu,axis = 0)/ np.sum(nu, axis = (0,1))

        self.means = np.einsum('k,kp->kp', 1.0 / np.sum(tau, axis=0), np.dot(tau.T, U))

        self.sigmas = np.einsum('k,klm->klm',
                                   1.0 / np.sum(tau, axis=0),
                                   np.stack([np.dot(np.einsum('i,ik->ik', tau[:, j], U - mu).T, U - mu) for j, mu in enumerate(self.means)]))
        return

    def fit(self, U, K=5,  Niter=50, Utest=None,
            pi1_0=None, A0=None, means0=None, sigmas0=None):
        """
        Computes the parameters of the Hidden Markov model with gaussian emission probabilities,
        using the EM algorithm.
        """
        n, p = U.shape

        if(pi1_0 == None):
            pi1_0 = (1.0 / n_classes) * np.ones(K)
        if(means0 == None):
            means0 = U[np.random.randint(n, size=K)]
        if (sigmas0 == None):
            sigmas0 = np.stack([np.identity(p) * (i + 1) for i in range(K)], axis = 0)
        if(A0 == None):
            A0 = np.identity(K) + np.ones((K, K))
            A0 /= np.sum(A0, axis=1)
        self.pi1 = pi1_0
        self.A = A0
        self.means = means0
        self.sigmas = sigmas0

        log_l = np.zeros(Niter) #  log likelihoods on training set
        if (Utest is not None):
            log_l_test = np.zeros(Niter) # log likelihoods on test set

        for i in range(Niter):
            # Alpha, Beta message passing
            la = self.alpha_log(U, self.pi1, self.A, self.means, self.sigmas) # Alpha messages in log scale
            lb = self.beta_log(U, self.pi1, self.A, self.means, self.sigmas) # Beta messages in log scale
            log_l[i] = (logsumexp(la + lb, axis=1))[0]
            if(Utest is not None):
                la_test = self.alpha_log(Utest, self.pi1, self.A, self.means, self.sigmas)
                lb_test = self.beta_log(Utest, self.pi1, self.A, self.means, self.sigmas)
                log_l_test[i] = (logsumexp(la_test + lb_test, axis=1))[0]

            # E step
            tau, nu = self.E_step(U, la, lb)

            # M step
            self.M_step(U, tau, nu)


        if(Utest is None):
            return self.pi1, self.A, self.means, self.sigmas, log_l

        else:
            return self.pi1, self.A, self.means, self.sigmas, log_l, log_l_test


    def predict_proba(self, U, logscale = True):
        """
        Computes the probabilities for each class of the t-th hidden variable qt in log scale
        """
        T, d = np.shape(U)
        K = np.shape(self.pi1)[0]

        log_alphas = self.alpha_log(U, self.pi1, self.A, self.means, self.sigmas)
        log_betas = self.beta_log(U, self.pi1, self.A, self.means, self.sigmas)
        log_proba = log_alphas + log_betas - logsumexp(log_alphas + log_betas, axis=1)

        if(logscale):
            return log_proba
        else:
            return np.exp(log_proba)

    def predict(self, U):
        """
        Predicts the class (hidden variable) of each observation in U
        """
        log_prob = self.predict_proba(U)
        return np.argmax(log_prob, axis=1)

    def viterbi(self, U):
        """
        Computes the most likely sequence recursively using the Viterbi algorithm
        """
        #initialization phase
        n = np.shape(self.A)[0]
        t = np.shape(U)[0]

        al = np.log(self.pi1[..., np.newaxis])


        #write coefficient in log scale
        factors = [gauss_log(U[0], mu, sigma) for (mu, sigma) in zip(self.means, self.sigmas)]
        al += np.reshape(factors, [n, 1])
        Alog = np.log(A)

        previous = np.reshape(np.zeros(4), [4, 1])

        #dynamic extension of alpha


        for k in range(1, t):

            old = np.reshape(al[:, -1], [1, n])
            factors = [gauss_log(U[k], mu, sigma) for (mu, sigma) in zip(self.means, self.sigmas)]
            factors = np.reshape(factors, [n, 1])

            tmp = Alog + old + factors
            new = np.reshape(np.max(tmp, 1), [n, 1])
            prev = np.where(tmp == new)[1]
            prev = np.reshape(prev, [n, 1])
            al = np.c_[al, new]
            previous = np.c_[previous, prev]

        previous = previous[:,1:]
        # construct the sequence with the highest probability
        lab = np.zeros(t)
        lab[t-1] = np.where(al[:,-1]==max(al[:,-1]))[0][0]
        for k in range(2,t+1):
            lab[-k]= previous[int(lab[1-k]),1-k]

        return lab
