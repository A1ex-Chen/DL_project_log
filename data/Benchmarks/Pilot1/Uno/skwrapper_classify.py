def classify(model, x, y, splits, features, threads=-1, prefix='', seed=0,
    class_weight=None):
    verify_path(prefix)
    model, name = get_model(model, threads, classify=True, seed=seed)
    train_scores, test_scores = [], []
    tests, preds = None, None
    probas = None
    best_model = None
    best_score = -np.Inf
    print('>', name)
    print('Cross validation:')
    for i, (train_index, test_index) in enumerate(splits):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.set_params(class_weight=class_weight)
        model.fit(x_train, y_train)
        train_score = model.score(x_train, y_train)
        test_score = model.score(x_test, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
        print('  fold {}/{}: score = {:.3f}  (train = {:.3f})'.format(i + 1,
            len(splits), test_score, train_score))
        if test_score > best_score:
            best_model = model
            best_score = test_score
        y_pred = model.predict(x_test)
        preds = np.concatenate((preds, y_pred)
            ) if preds is not None else y_pred
        tests = np.concatenate((tests, y_test)
            ) if tests is not None else y_test
        if hasattr(model, 'predict_proba'):
            probas_ = model.predict_proba(x_test)
            probas = np.concatenate((probas, probas_)
                ) if probas is not None else probas_
    uniques, counts = np.unique(tests, return_counts=True)
    average = 'binary' if len(uniques) <= 2 else 'weighted'
    roc_auc_score = None
    if probas is not None:
        fpr, tpr, thresholds = metrics.roc_curve(tests, probas[:, 1],
            pos_label=0)
        roc_auc_score = 1 - metrics.auc(fpr, tpr)
        roc_fname = '{}.{}.ROC'.format(prefix, name)
        if roc_auc_score:
            with open(roc_fname, 'w') as roc_file:
                roc_file.write('\t'.join(['Threshold', 'FPR', 'TPR']) + '\n')
                for ent in zip(thresholds, fpr, tpr):
                    roc_file.write('\t'.join('{0:.5f}'.format(x) for x in
                        list(ent)) + '\n')
    print('Average validation metrics:')
    naive_accuracy = max(counts) / len(tests)
    accuracy = np.sum(preds == tests) / len(tests)
    accuracy_gain = accuracy - naive_accuracy
    print(' ', score_format('accuracy_gain', accuracy_gain, signed=True))
    scores_fname = '{}.{}.scores'.format(prefix, name)
    metric_names = (
        'accuracy_score matthews_corrcoef f1_score precision_score recall_score log_loss'
        .split())
    with open(scores_fname, 'w') as scores_file:
        scores_file.write(score_format('accuracy_gain', accuracy_gain,
            signed=True, eol='\n'))
        for m in metric_names:
            s = None
            try:
                s = getattr(metrics, m)(tests, preds, average=average)
            except Exception:
                try:
                    s = getattr(metrics, m)(tests, preds)
                except Exception:
                    pass
            if s:
                print(' ', score_format(m, s))
                scores_file.write(score_format(m, s, eol='\n'))
        if roc_auc_score:
            print(' ', score_format('roc_auc_score', roc_auc_score))
            scores_file.write(score_format('roc_auc_score', roc_auc_score,
                eol='\n'))
        scores_file.write('\nModel:\n{}\n\n'.format(model))
    print()
    top_features = top_important_features(best_model, features)
    if top_features is not None:
        fea_fname = '{}.{}.features'.format(prefix, name)
        with open(fea_fname, 'w') as fea_file:
            fea_file.write(sprint_features(top_features))
    score = metrics.f1_score(tests, preds, average=average)
    return score
