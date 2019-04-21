import numpy as np

# Kmeans Implementation
# Two steps at each iteration: expectation and maximization

def assign_labels(X, centroids, k):
    """
    Assigns the closest centroid to each data point
    """
    Xr = np.stack([X for i in range(k)], axis = -1)

    distances = np.sum(np.square(Xr - centroids.T), axis = 1)

    return(np.argmin(distances, axis = -1), np.sum(np.min(distances, axis = -1)))

def update_centroids(X, labels, k):
    """
    Updates the centroids given the label assignation computer before
    """
    return(np.array([np.mean(X[labels == c], axis = 0) for c in range(k)]))

def KMeans(X, C0, Niter = 300):
    """
    Labels the input distribution into classes using the K-means algorithm

    Parameters
    ----------
        X : Array containing datapoints
        C0 : Array containing initial centroids
        Niter : Number of iterations to run
    Returns
    -------
        labels: label[i] is the label of the i-th observation X[i]
        centroids: final centroids
        errors: inertia at each iteration
    """

    k = np.shape(C0)[0] # number of clusters = number of initial centroids
    centroids = C0 # Centroids
    errors = np.zeros(Niter + 1) # Mean quadratic errors

    ## Compute initial labels

    labels, error = assign_labels(X, centroids, k)
    errors[0] = error

    ## Iterate
    for t in range(Niter):
        # Update centroids
        centroids = update_centroids(X, labels, k)

        # Assign labels using the new centroids, store the mean quadratic error
        labels, error = assign_labels(X, centroids, k)

        errors[t + 1] = error

    return labels, centroids, errors
