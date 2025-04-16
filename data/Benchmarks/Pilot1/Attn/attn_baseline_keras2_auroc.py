def auroc(y_true, y_pred):
    score = tf.py_func(lambda y_true, y_pred: roc_auc_score(y_true, y_pred,
        average='macro', sample_weight=None).astype('float32'), [y_true,
        y_pred], 'float32', stateful=False, name='sklearnAUC')
    return score
