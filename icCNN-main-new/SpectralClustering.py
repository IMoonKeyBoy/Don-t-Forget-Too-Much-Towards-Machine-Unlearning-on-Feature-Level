import sys

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
import numpy as np
import torch

import scipy
from scipy.sparse import csgraph
# from scipy.sparse.linalg import eigsh
from numpy import linalg as LA


def eigenDecomposition(A, plot=True, topK=10):
    """
    :param A: Affinity matrix
    :param plot: plots the sorted eigen values for visual inspection
    :return A tuple containing:
    - the optimal number of clusters by eigengap heuristic
    - all eigen values
    - all eigen vectors

    This method performs the eigen decomposition on a given affinity matrix,
    following the steps recommended in the paper:
    1. Construct the normalized affinity matrix: L = D−1/2ADˆ −1/2.
    2. Find the eigenvalues and their associated eigen vectors
    3. Identify the maximum gap which corresponds to the number of clusters
    by eigengap heuristic

    References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf
    """
    L = csgraph.laplacian(A, normed=True)
    #n_components = A.shape[0]

    # LM parameter : Eigenvalues with largest magnitude (eigs, eigsh), that is, largest eigenvalues in
    # the euclidean norm of complex numbers.
    #     eigenvalues, eigenvectors = eigsh(L, k=n_components, which="LM", sigma=1.0, maxiter=5000)
    eigenvalues, eigenvectors = LA.eig(L)

    # Identify the optimal number of clusters as the index corresponding
    # to the larger gap between eigen values
    print(eigenvalues.shape,eigenvalues)
    print(np.diff(eigenvalues))
    print(np.argsort(np.diff(eigenvalues)))
    print(np.argsort(np.diff(eigenvalues))[::-1])
    print(np.argsort(np.diff(eigenvalues))[::-1][:topK])
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:topK]
    nb_clusters = index_largest_gap + 1

    return nb_clusters, eigenvalues, eigenvectors

def spectral_clustering(similarity_matrix, n_cluster=8):
    W = similarity_matrix
    #k, _, _ = eigenDecomposition(similarity_matrix)
    #print(f'Optimal number of clusters {k}')
    #sys.exit(0)
    sz = W.shape[0]
    sp = SpectralClustering(n_clusters=n_cluster, affinity='precomputed', random_state=21)
    y_pred = sp.fit_predict(W)
    del W

    ground_true_matrix = np.zeros((sz, sz))
    loss_mask_num = []
    loss_mask_den = []
    for i in range(n_cluster):
        idx = np.where(y_pred == i)[0]
        cur_mask_num = np.zeros((sz, sz))
        cur_mask_den = np.zeros((sz, sz))
        for j in idx:
            ground_true_matrix[j][idx] = 1
            cur_mask_num[j][idx] = 1
            cur_mask_den[j][:] = 1
        loss_mask_num.append(np.expand_dims(cur_mask_num, 0))
        loss_mask_den.append(np.expand_dims(cur_mask_den, 0))
    loss_mask_num = np.concatenate(loss_mask_num, axis=0)
    loss_mask_den = np.concatenate(loss_mask_den, axis=0)
    return torch.from_numpy(ground_true_matrix).float().cuda(), torch.from_numpy(loss_mask_num).float().cuda(), torch.from_numpy(loss_mask_den).float().cuda()
