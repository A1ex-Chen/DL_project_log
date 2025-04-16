def evaluate_model(params, root_fname, nb_classes, Y_test, _Y_test,
    Y_predict, pos, total, score):
    threshold = 0.5
    Y_pred_int = (Y_predict[:, 0] < threshold).astype(np.int)
    Y_test_int = (Y_test[:, 0] < threshold).astype(np.int)
    print('creating table of predictions')
    f = open(params['save_path'] + root_fname + '.predictions.tsv', 'w')
    for index, row in _Y_test.iterrows():
        if row['AUC'] == 1:
            if Y_pred_int[index] == 1:
                call = 'TP'
            else:
                call = 'FN'
        if row['AUC'] == 0:
            if Y_pred_int[index] == 0:
                call = 'TN'
            else:
                call = 'FP'
        print(index, '\t', call, '\t', Y_pred_int[index], '\t', row['AUC'],
            '\t', row['Sample'], '\t', row['Drug1'], file=f)
    f.close()
    false_pos_rate, true_pos_rate, thresholds = roc_curve(Y_test[:, 0],
        Y_predict[:, 0])
    roc_auc = auc(false_pos_rate, true_pos_rate)
    auc_keras = roc_auc
    fpr_keras = false_pos_rate
    tpr_keras = true_pos_rate
    fname = params['save_path'] + root_fname + '.auroc.pdf'
    print('creating figure at ', fname)
    attnviz.plot_ROC(fpr_keras, tpr_keras, auc_keras, fname)
    fname = params['save_path'] + root_fname + '.auroc_zoom.pdf'
    print('creating figure at ', fname)
    attnviz.plot_ROC(fpr_keras, tpr_keras, auc_keras, fname, zoom=True)
    f1 = f1_score(Y_test_int, Y_pred_int)
    precision, recall, thresholds = precision_recall_curve(Y_test[:, 0],
        Y_predict[:, 0])
    pr_auc = auc(recall, precision)
    pr_keras = pr_auc
    precision_keras = precision
    recall_keras = recall
    print('f1=%.3f auroc=%.3f aucpr=%.3f' % (f1, auc_keras, pr_keras))
    fname = params['save_path'] + root_fname + '.aurpr.pdf'
    print('creating figure at ', fname)
    no_skill = len(Y_test_int[Y_test_int == 1]) / len(Y_test_int)
    attnviz.plot_RF(recall_keras, precision_keras, pr_keras, no_skill, fname)
    cnf_matrix = sklearn.metrics.confusion_matrix(Y_test_int, Y_pred_int)
    class_names = ['Non-Response', 'Response']
    fname = params['save_path'] + root_fname + '.confusion_without_norm.pdf'
    print('creating figure at ', fname)
    attnviz.plot_confusion_matrix(cnf_matrix, fname, classes=class_names,
        title='Confusion matrix, without normalization')
    fname = params['save_path'] + root_fname + '.confusion_with_norm.pdf'
    print('creating figure at ', fname)
    attnviz.plot_confusion_matrix(cnf_matrix, fname, classes=class_names,
        normalize=True, title='Normalized confusion matrix')
    print("""Examples:
    Total: {}
    Positive: {} ({:.2f}% of total)
"""
        .format(total, pos, 100 * pos / total))
    print(sklearn.metrics.roc_auc_score(Y_test_int, Y_pred_int))
    print(sklearn.metrics.balanced_accuracy_score(Y_test_int, Y_pred_int))
    print(sklearn.metrics.classification_report(Y_test_int, Y_pred_int))
    print(sklearn.metrics.confusion_matrix(Y_test_int, Y_pred_int))
    print('score')
    print(score)
    print('Test val_loss:', score[0])
    print('Test accuracy:', score[1])
