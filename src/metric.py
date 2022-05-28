from sklearn.metrics import confusion_matrix
import sklearn

def cal_metrics(gt_label, predicted_label, predictions):
    if(np.array(predicted_label).size != np.array(gt_label).size):
        print('wrong label size')
        print('wrong label size')
        print('wrong label size')
        print('wrong label size')
        print('wrong label size')
        print('wrong label size')


    fpr, tpr, _ = sklearn.metrics.roc_curve(gt_label, predictions)
    auc = 100 * sklearn.metrics.auc(fpr, tpr)
    tn, fp, fn, tp = confusion_matrix(gt_label, predicted_label).ravel()
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)

    sum_correct = 0
    num_labels = np.array(predicted_label).size

    for i in range(num_labels):
        gt_i = gt_label[i]
        predicted_i = predicted_label[i]
        if(gt_i == predicted_i):
            sum_correct += 1

    accuracy = sum_correct/num_labels

    return accuracy, auc, sensitivity, specificity


def cal_intersection_union(array_1, array_2):
    array_1 = np.array(array_1)
    array_2 = np.array(array_2)

    intersect = np.intersect1d(array_1, array_2)

    num_intersect = intersect.size
    if(num_intersect>0):
        print('************************************')
        print('oho, there is a intersection')
        print('************************************')

    union = np.union1d(array_1, array_2)
    num_union = union.size

    union_array = np.array(range(0, num_union))

    error = np.sum(union - union_array)

    if(error > 0):
        print('************************************')
        print('oops, wrong union')
        print('************************************')

