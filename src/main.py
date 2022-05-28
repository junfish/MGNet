import time
localtime = time.asctime( time.localtime(time.time()) )
print ("Local time :", localtime)
print('##$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$###############$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*****************')


########## Initialization

method = 'gcn'
n_class = 2
data_name = 'BP'

data, _, _, _, train_index_all, test_index_all, label_all = load_data_my_new(data_name, 0)
data = np.array(data, dtype= 'float32')
label_all = label_all[:,0]
N,n_views,M,F = data.shape

index_set_all = np.array(range(N))
R = 1

node_features = train_U_new(data, R)
U = node_features
data = data_projection(data, U)
data = np.array(data,dtype='float32')

num_layers = 2

process = 'concat'
view_pooling = 'max'

dropout = 0.1

iter = 0
kfold = 10

accuracy_total_sum = 0
auc_total_sum = 0
sensitivity_total_sum = 0
specificity_total_sum = 0

accuracy_total_record = []
auc_total_record = []
sensitivity_total_record = []
specificity_total_record = []

n_epoch = 3
batch_size = 6

########################## Main #####################
if __name__ == "__main__":
    for train_fold_chosen in range(kfold):
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('start training the model of fold:', train_fold_chosen)
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        _, _, indexes_set, labels_set, _, _, _ = load_data_my_new(data_name, train_fold_chosen)
        iter = 0
        indexes = indexes_set[0]
        labels = labels_set[0]

        accuracy_record = []
        param_record = {}

        train_idx, val_idx, _ = indexes
        train_idx = train_index_all[train_fold_chosen]
        train_labels, val_labels = label_all[train_idx], label_all[val_idx]
        train_data, val_data = data[train_idx, :, :, :], data[val_idx, :, :, :]
        test_index = test_index_all[train_fold_chosen]
        test_data, test_labels = data[test_index, :, :, :], label_all[test_index]
        cal_intersection_union(train_idx, test_index)

        for Out_dim1 in range(30, 50, 20):
            if (num_layers == 3):
                Input_dim = F
                o_dim1 = Out_dim1
                o_dim2 = o_dim1
                o_dim3 = n_class
                Output_dim = [o_dim1, o_dim2, o_dim3]
            elif (num_layers == 2):
                Input_dim = F
                o_dim1 = Out_dim1
                o_dim2 = n_class
                Output_dim = [o_dim1, o_dim2]
            shape_1 = [F, Out_dim1]
            shape_2 = [M * Out_dim1, 1800]
            shape_3 = [1800, n_class]
            shapes = []
            shapes.append(shape_1)
            shapes.append(shape_2)
            shapes.append(shape_3)
            weights, bias = initialize_bias_weights(shapes)

            for knn in range(6, 8, 8):
                A = obtain_normalized_adj(node_features, knn)
                if (isspmatrix(A)):
                    A = A.todense()
                A = np.array(A, dtype='float32')
                train_all_and_validate(train_data, train_labels, val_data, val_labels, test_data, test_labels, A,
                                       n_class, batch_size, 39, dropout, process, view_pooling)
                accuracy, auc, sensitivity, specificity = validate(test_data, test_labels, A, dropout, process,
                                                                   view_pooling)
                print('Out_dim1 = ', Out_dim1, 'knn = ', knn, 'validate accuracy = ', accuracy, 'batch_size =',
                      batch_size, 'dropout = ', dropout, 'process = ', process, 'view_pooling = ', view_pooling)

            print('test fold = ', train_fold_chosen, ' Out_dim1 = ', Out_dim1, 'knn = ', knn, 'accuracy = ', accuracy,
                  'auc = ', auc, 'sensitivity = ', sensitivity, 'specificity = ', specificity)

            del weights
            del bias

            accuracy_total_sum = accuracy_total_sum + accuracy
            auc_total_sum = auc_total_sum + auc
            sensitivity_total_sum = sensitivity_total_sum + sensitivity
            specificity_total_sum = specificity_total_sum + specificity

            accuracy_total_record.append(accuracy)
            auc_total_record.append(auc)
            sensitivity_total_record.append(sensitivity)
            specificity_total_record.append(specificity)

    avg_accuracy, avg_auc = accuracy_total_sum / kfold, auc_total_sum / kfold
    avg_sensitivity, avg_specificity = sensitivity_total_sum / kfold, specificity_total_sum / kfold

    print('%%%%%%%%%%%%%%%%% ')
    print(' All Stats ')
    print('%%%%%%%%%%%%%%%%% ')
    print('avg_acc:', avg_accuracy, 'avg_auc:', avg_auc, 'avg_sensi:', avg_sensitivity, 'avg_speci:', avg_specificity)

    print(np.array(accuracy_total_record))
    print(np.array(auc_total_record))
    print(np.array(sensitivity_total_record))
    print(np.array(specificity_total_record))


