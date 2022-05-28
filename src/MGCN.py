import tensorflow as tf


def feature_process(X, process='concat'):
    if (process == 'concat'):
        H, W = X.shape
        X_v = tf.reshape(X, [H * W, ])

    if (process == 'avg'):
        X_v = tf.reduce_mean(X, axis=0)

    if (process == 'max'):
        X_v = tf.reduce_max(X, axis=0)

    return X_v


def Graphconvolution_all(A, x, dropout, process):
    if (isspmatrix(A)):
        A = A.todense()

    num_layers = len(weights) - 2
    if (len(weights) != len(bias)):
        print('length error of initialization')

    tf.random.set_seed(0)
    input = tf.nn.dropout(x, dropout)
    # input = x

    for i in range(num_layers):
        output = tf.matmul(input, weights[i])
        output = tf.matmul(A, output)
        output = output + bias[i]

        output = tf.nn.relu(output)
        input = output

    output = feature_process(output, process)

    return output


def view_pooling_func(X, view_pooling):
    if (view_pooling == 'avg'):
        X_pooling = tf.reduce_mean(X, axis=0)

    if (view_pooling == 'max'):
        X_pooling = tf.reduce_max(X, axis=0)

    if (view_pooling == 'concat'):
        [V, F] = X.shape
        X_pooling = tf.reshape(X, [1, V * F])

    return X_pooling


def MV_model_infer(inputs_all, A, dropout, process, view_pooling):
    N, V, M, F = inputs_all.shape
    outputs_all = []
    for i in range(N):
        outputs_all_v = []
        for v in range(V):
            input_i_v = inputs_all[i, v, :, :]
            output_i = Graphconvolution_all(A, input_i_v, dropout, process)
            outputs_all_v.append(output_i)

        output_feature_all_v = tf.convert_to_tensor(outputs_all_v)
        View_feature_v = view_pooling_func(output_feature_all_v, view_pooling)
        # View_feature_v = 0.018*output_feature_all_v[0,:] + 0.999 * output_feature_all_v[1,:] ## For HIV
        # View_feature_v = 0.0089*output_feature_all_v[0,:] + 0.999 * output_feature_all_v[1,:] ## For BP

        outputs_all.append(View_feature_v)

    outputs_all = tf.convert_to_tensor(outputs_all)

    # if(process == 'concat'):
    #   _,F_out = outputs_all.shape
    #   outputs_all = tf.reshape(outputs_all, [int(N * M), int(F_out/M)])
    #   outputs_all = tf.nn.l2_normalize(outputs_all, axis=1, epsilon=1e-12, name=None)
    #   outputs_all = tf.reshape(outputs_all, [int(N), int(F_out)])
    outputs_all = tf.nn.dropout(outputs_all, dropout)

    outputs_all = tf.matmul(outputs_all, weights[-2]) + bias[-2]
    outputs_all = tf.nn.relu(outputs_all)

    logits = tf.matmul(outputs_all, weights[-1]) + bias[-1]

    return logits


def MV_model_infer_concat(inputs_all, A, dropout, process, view_pooling):
    N, V, M, F = inputs_all.shape
    for i in range(N):
        for v in range(V):
            input_i_v = inputs_all[i, v, :, :]
            output_i = Graphconvolution_all(A, input_i_v, dropout, process)
            output_i = tf.expand_dims(output_i, 0)
            if (v == 0):
                outputs_all_v = output_i
            else:
                outputs_all_v = tf.concat([outputs_all_v, output_i], axis=0)

        View_feature_v = view_pooling_func(outputs_all_v, view_pooling)
        View_feature_v = tf.expand_dims(View_feature_v, 0)

        if (i == 0):
            outputs_all = View_feature_v
        else:
            outputs_all = tf.concat([outputs_all, View_feature_v], axis=0)

    outputs_all = tf.nn.dropout(outputs_all, dropout)

    logits = tf.matmul(outputs_all, weights[-1]) + bias[-1]

    return logits
