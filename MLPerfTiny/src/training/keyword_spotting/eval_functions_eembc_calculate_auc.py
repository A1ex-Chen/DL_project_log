def calculate_auc(y_pred, labels, classes, name):
    n_classes = len(classes)
    thresholds = np.arange(0.0, 1.01, 0.01)
    fpr = np.zeros([n_classes, len(thresholds)])
    tpr = np.zeros([n_classes, len(thresholds)])
    roc_auc = np.zeros(n_classes)
    for class_item in range(n_classes):
        all_positives = sum(labels == class_item)
        all_negatives = len(labels) - all_positives
        for threshold_item in range(1, len(thresholds)):
            threshold = thresholds[threshold_item]
            false_positives = 0
            true_positives = 0
            for i in range(len(y_pred)):
                if y_pred[i, class_item] > threshold:
                    if labels[i] == class_item:
                        true_positives += 1
                    else:
                        false_positives += 1
            fpr[class_item, threshold_item] = false_positives / float(
                all_negatives)
            tpr[class_item, threshold_item] = true_positives / float(
                all_positives)
        fpr[class_item, 0] = 1
        tpr[class_item, 0] = 1
        for threshold_item in range(len(thresholds) - 1):
            roc_auc[class_item] += 0.5 * (tpr[class_item, threshold_item] +
                tpr[class_item, threshold_item + 1]) * (fpr[class_item,
                threshold_item] - fpr[class_item, threshold_item + 1])
    roc_auc_avg = np.mean(roc_auc)
    print(f'Simplified average roc_auc = {roc_auc_avg:.3f}')
    plt.figure()
    for class_item in range(n_classes):
        plt.plot(fpr[class_item, :], tpr[class_item, :], label=
            f'auc: {roc_auc[class_item]:0.3f} ({classes[class_item]})')
    plt.xlim([0.0, 0.1])
    plt.ylim([0.5, 1.0])
    plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC: ' + name)
    plt.grid(which='major')
    plt.show(block=False)
    return roc_auc
