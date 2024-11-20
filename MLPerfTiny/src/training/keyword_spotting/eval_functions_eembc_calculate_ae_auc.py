def calculate_ae_auc(y_pred, y_true, name):
    thresholds = np.amin(y_pred) + np.arange(0.0, 1.01, 0.01) * (np.amax(
        y_pred) - np.amin(y_pred))
    roc_auc = 0
    n_normal = np.sum(y_true == 0)
    tpr = np.zeros(len(thresholds))
    fpr = np.zeros(len(thresholds))
    for threshold_item in range(1, len(thresholds)):
        threshold = thresholds[threshold_item]
        y_pred_binary = (y_pred > threshold).astype(int)
        tpr[threshold_item] = np.sum(y_pred_binary[n_normal:] == 1) / float(
            len(y_true) - n_normal)
        fpr[threshold_item] = np.sum(y_pred_binary[0:n_normal] == 1) / float(
            n_normal)
    fpr[0] = 1
    tpr[0] = 1
    for threshold_item in range(len(thresholds) - 1):
        roc_auc += 0.5 * (tpr[threshold_item] + tpr[threshold_item + 1]) * (fpr
            [threshold_item] - fpr[threshold_item + 1])
    print(f'Simplified roc_auc = {roc_auc:.3f}')
    plt.figure()
    plt.plot(tpr, fpr, label=f'auc: {roc_auc:0.3f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend(loc='lower right')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC: ' + name)
    plt.grid(which='major')
    plt.show(block=False)
    return roc_auc
