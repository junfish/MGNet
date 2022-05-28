import sklearn, sklearn.datasets
import sklearn.naive_bayes, sklearn.linear_model, sklearn.svm, sklearn.neighbors, sklearn.ensemble
import matplotlib.pyplot as plt
import scipy.sparse
import numpy as np
import time, re
import pickle as pkl
import pandas as pd
import scipy.io as sio
from scipy import *
import random
from sklearn.model_selection import StratifiedKFold


def data_projection(data, U):
    U_trans = np.transpose(U)
    M_U, _ = U_trans.shape
    N, V, M, F = data.shape
    new_data = np.zeros([N, V, M_U, F])
    for i in range(N):
        for v in range(V):
            data_iv = data[i, v, :, :]
            data_iv_trans = np.matmul(U_trans, data_iv)
            # data_iv_trans = np.matmul(data_iv_trans, U)
            new_data[i, v, :, :] = data_iv_trans

    return new_data


def convert_minus_one_2_zero(label):
    for i in range(len(label)):
        if (label[i] < 0):
            label[i] = 0
    return label


def load_data(f):
    k = 0
    input_data = {}

    for line in f:
        input_data[k] = np.int32(line.split())
        k = k + 1

    return input_data


def load_train_index(f):
    train_index = []
    k = 0
    for line in f:
        train_index.append(np.int32(line))

    return train_index


def load_data_my_new(data_name, train_fold_chosen):
    # load data from google drive
    if (data_name == 'BP'):
        x = scipy.io.loadmat('BP.mat')
        X_normalize = x['X_normalize']
        label = x['label']
    elif (data_name == 'HIV'):
        x = scipy.io.loadmat('HIV.mat')
        X_normalize = x['X']
        label = x['label']
    elif (data_name == 'PPMI'):
        x = scipy.io.loadmat('PPMI.mat')
        X_normalize = x['X']
        label = x['label']

    # reshape data_size to N,V,M,F
    N_subject = X_normalize.size
    X_1 = X_normalize[0][0]
    M, F, V = X_1.shape
    print('loading data:', data_name, 'of size:', N_subject, V, M, F)
    data = np.zeros([N_subject, V, M, F])
    for i in range(N_subject):
        X_i = X_normalize[i][0]
        for j in range(V):
            X_ij = X_i[:, :, j]
            data[i, j, :, :] = X_ij

    labels = np.array(label)
    if (data_name == 'BP'):
        f_test = open('results/BP_divide_test_index.txt', 'r')
        test_idx = load_data(f_test)
        f_train = open('results/BP_divide_train_index.txt', 'r')
        train_idx = load_data(f_train)
    elif (data_name == 'HIV'):
        f_test = open('results/HIV_divide_test_index.txt', 'r')
        test_idx = load_data(f_test)
        f_train = open('results/HIV_divide_train_index.txt', 'r')
        train_idx = load_data(f_train)
    elif (data_name == 'PPMI'):
        f_test = open('results/PPMI_divide_test_index.txt', 'r')
        test_idx = load_data(f_test)
        f_train = open('results/PPMI_divide_train_index.txt', 'r')
        train_idx = load_data(f_train)

    f_test.close()
    f_train.close()

    train_set_ratio = 0.8
    train_chosen_idx = train_idx[train_fold_chosen]
    data_size = train_chosen_idx.size
    train_data_size = np.floor(data_size * train_set_ratio)
    train_index = random.sample(list(train_chosen_idx), int(train_data_size))
    test_index = np.setdiff1d(train_chosen_idx, train_index)

    labels_set = list()
    indexes_set = list()

    train_label, test_label = labels[train_index], labels[test_index]
    val_label, val_index = test_label, test_index
    train_label = convert_minus_one_2_zero(train_label)
    val_label = convert_minus_one_2_zero(val_label)
    test_label = convert_minus_one_2_zero(test_label)

    train_label = np.array(train_label).flatten()
    val_label = np.array(val_label).flatten()
    test_label = np.array(test_label).flatten()

    train_index = np.array(train_index).flatten()
    val_index = np.array(val_index).flatten()
    test_index = np.array(test_index).flatten()

    labels_set.append((train_label, val_label, test_label))
    indexes_set.append((train_index, val_index, test_index))

    train_index_all = train_idx
    test_index_all = test_idx

    subj = 0
    return data, subj, indexes_set, labels_set, train_index_all, test_index_all, convert_minus_one_2_zero(labels)

