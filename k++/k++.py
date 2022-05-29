# K-Means++ Optimal Data Clustering Algorithm v.0.0.1
#
#        C,S = compute(N,k)
#
#        N - # of observations, k - # of clusters
#
#        The worst-case complexity of the K-Means++ procedure:
#
#                   p = O(k^2ndi + nd)
#
#                   An Example: n = 10^2, d = 2, k = 3, i = 3
#
#                               p = O(3^2 * 10^2 * 2 * 3 + 10^2 * 2) = O(5400)
#
#   GNU Public License (C) 2021 Arthur V. Ratz
#-----------------------------------------------------------------------------------

import math
import time
import random
import numpy as np
import pandas as pd
import numpy.linalg as lin
import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_blobs

def gen_dataset(n,d,k):
    # Generate an arbitrary dataset X as a matrix (n x d) of random elements [0;1)
    return np.random.random_sample((n,d))

def gen_dataset_synth(n,d,k):
    # Generate a synthetic dataset X as a distribution of n-points
    # having d-features, arranged into k-clusters with random standard deviation [0.75;1.0)
    return make_blobs(n_samples=n, n_features=d, \
        centers=k, cluster_std=np.random.uniform(0.75, 1.0), random_state=1)[0]

def exists(E,i):
    # return 'True' if the point i exists
    # in the array E, and 'False' unless otherwise
    return 0 < np.shape(np.array( \
        [ e for e in np.array(E) if (e == i).all() ]))[0]

def eucld(i1,i2):
    # Compute the squared Euclidean distance d=|i1-i2|^2 as the sum of squared
    # distances between points i1 and i2, at each dimension
    return np.sum(np.array([ \
        math.pow(i1 - i2, 2.0) \
          for i1, i2 in zip(i1, i2) ]))

def initialize(X):
    # Get the random centroid c0
    c0 = np.random.randint(0, np.shape(X)[0] - 1) + 1

    # Compute the distance from centroid c0 to each point in X
    c0_d = np.array([ eucld(X[c0], x) for x in X ])

    # Get the centroid c1's as one of the points in X,
    # having the maximum distance to the centroid c0
    c1 = np.where(c0_d >= np.max(c0_d))[0][0]

    return np.array([c0,c1]) # Return the indexes of c0 and c1

def compute(X,k):
    X = np.array(X)    # X - an input dataset of n-observations
    C = initialize(X)   # C - an initial set of centroids

    # Perform the dataset clustering iteratively,
    # until the resultant set of k-clusters has been computed

    while True:
        S = np.empty(0)  # S - a set of newly built clusters

        # For each observation x[t] in X, do the following:
        for t in range(np.shape(X)[0]):
            # Check if the observation x[t] has already been
            # selected as one of the new centroids
            if exists(C, t) == False:
                # If not, compute the distance from
                # the observation x[t] to each of the existing centroids in C
                cn_ds = np.array([ eucld(X[t], X[c]) for c in C ])
                # Get the centroid c[r] for which the distance to x[t] is the smallest
                cn_min_di = np.where(cn_ds == np.min(cn_ds))[0][0]

                # Assign the observation x[t] to the new cluster s[r], appending
                # the observation x[t]'s and centroid c[r]'s indexes to the set S
                S = np.append(S, { 'c': cn_min_di, 'i': t, 'd': cn_ds[cn_min_di] })

        # Terminate the clustering process, if the number of centroids
        # in C is equal to the total number of clusters k, initially specified.

        # Otherwise, compute the next centroid c[r] in C

        if np.shape(C)[0] >= k: break

        # Get the distances |x-c| from the observations
        # accross all clusters in S to each of the centroids in C
        cn_ds = np.array([s['d'] for s in S ])

        # Compute the index of an observation, for which
        # the distance to one of the centroids in C is the largest
        cn_max_ci = np.where(cn_ds == np.max(cn_ds))[0][0]

        # Append the index of a new centroid c[r] to the set C
        C = np.append(C, S[cn_max_ci]['i'])

    return C,S
