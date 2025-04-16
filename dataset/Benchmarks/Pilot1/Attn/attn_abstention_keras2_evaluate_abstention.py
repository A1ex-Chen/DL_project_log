def evaluate_abstention(params, root_fname, nb_classes, Y_test, _Y_test,
    Y_predict, pos, total, score):
    Y_pred_int = np.argmax(Y_predict, axis=1).astype(np.int)
    Y_test_int = np.argmax(Y_test, axis=1).astype(np.int)
    Y_pred_abs = (Y_pred_int == nb_classes).astype(np.int)
    abs0 = 0
    abs1 = 0
    print('creating table of predictions (with abstention)')
    f = open(params['save_path'] + root_fname + '.predictions.tsv', 'w')
    for index, row in _Y_test.iterrows():
        if row['AUC'] == 1:
            if Y_pred_abs[index] == 1:
                call = 'ABS1'
                abs1 += 1
            elif Y_pred_int[index] == 1:
                call = 'TP'
            else:
                call = 'FN'
        if row['AUC'] == 0:
            if Y_pred_abs[index] == 1:
                call = 'ABS0'
                abs0 += 1
            elif Y_pred_int[index] == 0:
                call = 'TN'
            else:
                call = 'FP'
        print(index, '\t', call, '\t', Y_pred_int[index], '\t', row['AUC'],
            '\t', Y_pred_abs[index], '\t', row['Sample'], '\t', row['Drug1'
            ], file=f)
    f.close()
    index_pred_noabs = Y_pred_int < nb_classes
    Y_test_noabs = Y_test[index_pred_noabs, :2]
    Y_test_int_noabs = Y_test_int[index_pred_noabs]
    Y_pred_noabs = Y_predict[index_pred_noabs, :2] / np.sum(Y_predict[
        index_pred_noabs, :2], axis=1, keepdims=True)
    Y_pred_int_noabs = Y_pred_int[index_pred_noabs]
    false_pos_rate, true_pos_rate, thresholds = roc_curve(Y_test_noabs[:, 0
        ], Y_pred_noabs[:, 0])
    roc_auc = auc(false_pos_rate, true_pos_rate)
    auc_keras = roc_auc
    fpr_keras = false_pos_rate
    tpr_keras = true_pos_rate
    fname = params['save_path'] + root_fname + '.auroc.pdf'
    print('creating figure at ', fname)
    add_lbl = ' (after removing abstained samples) '
    attnviz.plot_ROC(fpr_keras, tpr_keras, auc_keras, fname, xlabel_add=
        add_lbl, ylabel_add=add_lbl)
    fname = params['save_path'] + root_fname + '.auroc_zoom.pdf'
    print('creating figure at ', fname)
    attnviz.plot_ROC(fpr_keras, tpr_keras, auc_keras, fname, xlabel_add=
        add_lbl, ylabel_add=add_lbl, zoom=True)
    f1 = f1_score(Y_test_int_noabs, Y_pred_int_noabs)
    precision, recall, thresholds = precision_recall_curve(Y_test_noabs[:, 
        0], Y_pred_noabs[:, 0])
    pr_auc = auc(recall, precision)
    pr_keras = pr_auc
    precision_keras = precision
    recall_keras = recall
    print('f1=%.3f auroc=%.3f aucpr=%.3f' % (f1, auc_keras, pr_keras))
    fname = params['save_path'] + root_fname + '.aurpr.pdf'
    print('creating figure at ', fname)
    no_skill = len(Y_test_int_noabs[Y_test_int_noabs == 1]) / len(
        Y_test_int_noabs)
    attnviz.plot_RF(recall_keras, precision_keras, pr_keras, no_skill,
        fname, xlabel_add=add_lbl, ylabel_add=add_lbl)
    cnf_matrix = sklearn.metrics.confusion_matrix(Y_test_int, Y_pred_int)
    class_names = ['Non-Response', 'Response', 'Abstain']
    fname = params['save_path'] + root_fname + '.confusion_without_norm.pdf'
    attnviz.plot_confusion_matrix(cnf_matrix, fname, classes=class_names,
        title='Confusion matrix, without normalization')
    print(
        'NOTE: Confusion matrix above has zeros in the last row since the ground-truth does not include samples in the abstaining class.'
        )
    fname = params['save_path'] + root_fname + '.confusion_with_norm.pdf'
    attnviz.plot_confusion_matrix(cnf_matrix, fname, classes=class_names,
        normalize=True, title='Normalized confusion matrix')
    print(
        'NOTE: Normalized confusion matrix above has NaNs in the last row since the ground-truth does not include samples in the abstaining class.'
        )
    print("""Examples:
    Total: {}
    Positive: {} ({:.2f}% of total)
"""
        .format(total, pos, 100 * pos / total))
    total_pred = Y_pred_int_noabs.shape[0]
    print(
        """Abstention (in prediction):  Label0: {} ({:.2f}% of total pred)
 Label1: {} ({:.2f}% of total pred)
"""
        .format(abs0, 100 * abs0 / total_pred, abs1, 100 * abs1 / total_pred))
    print(sklearn.metrics.roc_auc_score(Y_test_int_noabs, Y_pred_int_noabs))
    print(sklearn.metrics.balanced_accuracy_score(Y_test_int_noabs,
        Y_pred_int_noabs))
    print(sklearn.metrics.classification_report(Y_test_int_noabs,
        Y_pred_int_noabs))
    print(sklearn.metrics.confusion_matrix(Y_test_int_noabs, Y_pred_int_noabs))
    print('Score: ', score)
    print('Test val_loss (not abstained samples):', score[0])
    print('Test accuracy (not abstained samples):', score[1])
