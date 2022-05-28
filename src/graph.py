### re-implement graph.py to construct ajacency matrix A and normalized laplacian matrix L_hat

import sklearn.metrics
import sklearn.neighbors
import scipy
import numpy as np
from scipy.sparse import rand
import scipy.sparse as sp
from scipy.sparse import isspmatrix
from scipy.sparse.linalg.eigen.arpack import eigsh

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized


### construct ajacency matrix
def compute_dist(graph, k, metric):
    dist = scipy.spatial.distance.pdist(graph, metric)
    dist_square = scipy.spatial.distance.squareform(dist)
    id_dist = np.argsort(dist_square)[:, :k + 1]
    dist_square.sort()
    dist_square = dist_square[:, :k + 1]
    return dist_square, id_dist


def build_ajacency(dist, idx):
    M, k = dist.shape
    sigma = np.mean(dist[:, -1]) ** 2
    dist = np.exp(-dist / sigma)

    # construct sparse matrix
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M * k)
    V = dist.reshape(M * k)
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))
    W.setdiag(0)
    # construct symmetric matrix
    bigger_index = W.T > W
    # W = W - W.multiply(bigger_index) + W.T.multiply(bigger_index)
    W = W + W.T.multiply(bigger_index)
    assert (type(W)) is scipy.sparse.csr.csr_matrix

    return W


def build_laplacian(W, normalized=True):
    d = W.sum(axis=0)
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D
        # np.savetxt(r'results/PPMI/PPMI_Laplacian.txt',L.todense(),fmt="%f")
        # print('I am writing！！！！')
        # L = scipy.sparse.rand(d.size, d.size, density=0.25, format="csr", random_state=42)

    # assert(type(L)) is scipy.sparse.csr_matrix
    return L


def rescale_L(L, l_max):
    L = 2 / l_max * L
    I = scipy.sparse.identity(L.shape[0], format='csr', dtype=L.dtype)
    L_hat = L - I
    return L_hat


def obtain_normalized_adj(node_features, k, metric='euclidean'):
    dist, idx = compute_dist(node_features, k, metric='euclidean')
    A = build_ajacency(dist, idx).astype(np.float32)
    I = scipy.sparse.eye(A.shape[0], dtype=A.dtype)
    A_hat = A + I

    d = A_hat.sum(axis=0)
    d += np.spacing(np.array(0, A_hat.dtype))
    d = 1 / np.sqrt(d)
    D = scipy.sparse.diags(d.A.squeeze(), 0)
    normalized_A = D * A_hat * D

    return normalized_A


def chebyshev_polynomials(adj, k):
    """
    Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation).
    """
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)

if __name__ == "__main__":
    node_features = np.random.rand(4,4)
    k = 2
    normalized_A = obtain_normalized_adj(node_features,k)## Either code works fine to obtain the normalzed A
    normalized_A2 = preprocess_adj(A)

