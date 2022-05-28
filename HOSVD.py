import tensorly as tl
import time
from  scipy import *

def train_U_with_noise(data, R, train_idx):
    data_chosen = data[train_idx,:,:,:]
    X1 = tl.unfold(data_chosen, mode=1)
    B = np.matmul(X1,X1.transpose())
    U, S, V = np.linalg.svd(B, full_matrices=True)
    S = np.sqrt(S)
    sum_all_S = np.sum(S)
    len_S = len(S)
    sum_part_S = 0
    for i in range(len_S):
        sum_part_S = sum_part_S + S[i]
        if(sum_part_S>sum_all_S*R):
            break
    U = U[:,0:i]
    return U

def train_U_new(data, R):
    N,V,M,F = data.shape
    sum_X = 0
    for i in range(N):
        for v in range(V):
            X_iv = data[i,v,:,:]
            sum_X = sum_X + np.matmul(X_iv, X_iv.transpose())

    U, S, V = np.linalg.svd(sum_X, full_matrices=True)
    S = np.sqrt(S)
    sum_all_S = np.sum(S)
    len_S = len(S)
    sum_part_S = 0
    for i in range(len_S):
        sum_part_S = sum_part_S + S[i]
        if(sum_part_S>sum_all_S*R):
            break
    if(R == 1):
        U = U
    else:
        U = U[:,0:i]

    return U

def train_U(data_name, R):
    if(data_name == 'BP'):
        x = scipy.io.loadmat('BP.mat')
        X_normalize = x['X_normalize']

    elif(data_name == 'HIV'):
        x = scipy.io.loadmat('HIV.mat')
        X_normalize = x['X']

    elif(data_name == 'PPMI'):
        x = scipy.io.loadmat('PPMI.mat')
        X_normalize = x['X']

    N = X_normalize.size
    X_1 = X_normalize[0][0]
    M,F,V = X_1.shape
    # print('loading data:',data_name,'of size:',N,V,M,F)
    data = np.zeros([N,M,F,V])
    for i in range(N):
        X_i = X_normalize[i][0]
        for j in range(V):
            X_ij = X_i[:,:,j]
            data[i,:,:,j] = X_ij

    data_select = data
    X1 = tl.unfold(data_select, mode=1)
    B = np.matmul(X1,X1.transpose())
    U, S, V = np.linalg.svd(B, full_matrices=True)
    S = np.sqrt(S)
    sum_all_S = np.sum(S)
    len_S = len(S)
    sum_part_S = 0
    for i in range(len_S):
        sum_part_S = sum_part_S + S[i]
        if(sum_part_S>sum_all_S*R):
            break
    if(R == 1):
        U = U
    else:
        U = U[:,0:i]
    return U

