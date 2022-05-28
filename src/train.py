def get_loss(logits, gt_labels):
    gt_labels = tf.stop_gradient(gt_labels)
    return tf.nn.softmax_cross_entropy_with_logits(gt_labels, logits)


optimizer = tf.optimizers.Adam(learning_rate=0.001)


def batch_train(batch_inputs, batch_labels, A, dropout, process, view_pooling):
    with tf.GradientTape() as tape:
        logits = MV_model_infer(batch_inputs, A, dropout, process, view_pooling)
        # logits = model_infer(inputs_all, A, n_class, dropout, process)
        current_loss = get_loss(logits, batch_labels)

        trainable_variables = weights + bias

        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in trainable_variables])
        lossL2_decay = tf.reduce_mean(current_loss) + 0.001 * lossL2

        grads = tape.gradient(lossL2_decay, trainable_variables)
        # grads = tape.gradient(current_loss, trainable_variables)

        # Update the weights
        optimizer.apply_gradients(zip(grads, trainable_variables))

    avg_loss = tf.reduce_mean(current_loss)

    return avg_loss


def train_all(Inputs_train, train_labels, A, n_class, batch_size, n_epoch, dropout, process, view_pooling):
    N_train = np.array(train_labels).size
    num_batches_per_epoch = int((N_train - 1) / batch_size) + 1
    ### Train
    for iter in range(n_epoch):
        epoch_loss = 0
        idx_all = np.random.permutation(N_train)
        start_idx = 0
        for i in range(num_batches_per_epoch):
            end_idx = start_idx + batch_size
            if (end_idx > N_train):
                end_idx = N_train
            selected_idx = idx_all[start_idx:end_idx]

            train_label_select = train_labels[selected_idx]
            Inputs_select = Inputs_train[selected_idx, :, :, :]
            train_label_select_one_hot = tf.one_hot(train_label_select, depth=n_class)
            avg_loss = batch_train(Inputs_select, train_label_select_one_hot, A, dropout, process, view_pooling)
            epoch_loss += avg_loss

            start_idx += batch_size

        if (iter % 10 == 0 or iter == n_epoch - 1):
            print('epoch = ', iter, 'loss = ', epoch_loss)


def train_all_and_validate(Inputs_train, train_labels, val_data, val_labels, test_data, test_labels, A, n_class,
                           batch_size, n_epoch, dropout, process, view_pooling):
    N_train = np.array(train_labels).size
    num_batches_per_epoch = int((N_train - 1) / batch_size) + 1
    ### Train
    for iter in range(n_epoch):
        epoch_loss = 0
        idx_all = np.random.permutation(N_train)
        start_idx = 0
        for i in range(num_batches_per_epoch):
            end_idx = start_idx + batch_size
            if (end_idx > N_train):
                end_idx = N_train
            selected_idx = idx_all[start_idx:end_idx]

            train_label_select = train_labels[selected_idx]
            Inputs_select = Inputs_train[selected_idx, :, :, :]
            train_label_select_one_hot = tf.one_hot(train_label_select, depth=n_class)
            avg_loss = batch_train(Inputs_select, train_label_select_one_hot, A, dropout, process, view_pooling)
            epoch_loss += avg_loss

            start_idx += batch_size

        accuracy_val, auc_val, sensitivity, specificity = validate(val_data, val_labels, A, dropout, process,
                                                                   view_pooling)

        accuracy_test, auc_test, sensitivity, specificity = validate(test_data, test_labels, A, dropout, process,
                                                                     view_pooling)

        print('epoch = ', iter, 'loss = ', epoch_loss, 'val accuracy = ', accuracy_val, 'test accuracy = ',
              accuracy_test)


def validate(Inputs_val, val_labels, A, dropout, process, view_pooling):
    logits = MV_model_infer(Inputs_val, A, dropout, process, view_pooling)
    predictions = tf.nn.softmax(logits)
    predicted_labels = np.argmax(predictions, axis=1)
    predictions_auc = np.max(predictions, axis=1)

    # print(predictions)
    # print(predicted_labels)
    # print(val_labels)

    accuracy, auc, sensitivity, specificity = cal_metrics(val_labels, predicted_labels, predicted_labels)

    return accuracy, auc, sensitivity, specificity

