import numpy as np


def reshape_data(data):
    N, M, F, V = data.shape
    data_reshape = np.zeros([N, V, M, F])
    for i in range(N):
        X_i = data[i, :, :, :]
        for j in range(V):
            X_ij = X_i[:, :, j]
            data_reshape[i, j, :, :] = X_ij

    return data_reshape


def reshape_data_back(data):
    N, V, M, F = data.shape
    data_reshape = np.zeros([N, M, F, V])
    for i in range(N):
        X_i = data[i, :, :, :]
        for v in range(V):
            X_ij = X_i[v, :, :]
            data_reshape[i, :, :, v] = X_ij

    return data_reshape


def load_data_only(data_name):
    # load data from google drive
    if (data_name == 'BP'):
        x = scipy.io.loadmat('BP.mat')
        X_normalize = x['X_normalize']
    elif (data_name == 'HIV'):
        x = scipy.io.loadmat('HIV.mat')
        X_normalize = x['X']
    elif (data_name == 'PPMI'):
        x = scipy.io.loadmat('PPMI.mat')
        X_normalize = x['X']

    N = X_normalize.size
    X_1 = X_normalize[0][0]
    M, F, V = X_1.shape
    data = np.zeros([N, M, F, V])
    for i in range(N):
        X_i = X_normalize[i][0]
        for j in range(V):
            X_ij = X_i[:, :, j]
            data[i, :, :, j] = X_ij

    return data


def add_noise(data, view, sigma):
    [N, M, F, V] = data.shape;
    X = data.copy()
    if (view < V):
        X_v = data[:, :, :, view]
        s = np.random.normal(0, sigma, [N, M, F])
        X_corrupted = X_v + np.array(s)
        # print(X_corrupted - X_v)
        X[:, :, :, view] = X_corrupted

    elif (view == V):
        print('adding noise to both views')
        X_v1 = data[:, :, :, 0]
        X_v2 = data[:, :, :, 1]

        s1 = np.random.normal(0, sigma, [N, M, F])
        X_corrupted1 = X_v1 + np.array(s1)

        s2 = np.random.normal(0, sigma, [N, M, F])
        X_corrupted2 = X_v2 + np.array(s2)

        X[:, :, :, 0] = X_corrupted1
        X[:, :, :, 1] = X_corrupted2

    return X
if __name__ == "__main__":
    X = np.random.rand(3,3,2,2)
    X_noise = add_noise(X, 0, 0.1)
    print(X - X_noise)

