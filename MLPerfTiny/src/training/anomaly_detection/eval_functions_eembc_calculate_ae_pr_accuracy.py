def calculate_ae_pr_accuracy(y_pred, y_true):
    thresholds = np.amin(y_pred) + np.arange(0.0, 1.0, 0.01) * (np.amax(
        y_pred) - np.amin(y_pred))
    accuracy = 0
    n_normal = np.sum(y_true == 0)
    precision = np.zeros(len(thresholds))
    recall = np.zeros(len(thresholds))
    for threshold_item in range(len(thresholds)):
        threshold = thresholds[threshold_item]
        y_pred_binary = (y_pred > threshold).astype(int)
        true_negative = np.sum(y_pred_binary[0:n_normal] == 0)
        false_positive = np.sum(y_pred_binary[0:n_normal] == 1)
        true_positive = np.sum(y_pred_binary[n_normal:] == 1)
        false_negative = np.sum(y_pred_binary[n_normal:] == 0)
        precision[threshold_item] = true_positive / (true_positive +
            false_positive)
        recall[threshold_item] = true_positive / (true_positive +
            false_negative)
        accuracy_tmp = 100 * (precision[threshold_item] + recall[
            threshold_item]) / 2
        if accuracy_tmp > accuracy:
            accuracy = accuracy_tmp
    print(f'Precision/recall accuracy = {accuracy:2.1f}')
    plt.figure()
    plt.plot(recall, precision)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall')
    plt.grid(which='major')
    plt.show(block=False)
    return accuracy
