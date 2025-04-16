def regress(model, x, y, splits, features, threads=-1, prefix='', seed=0):
    verify_path(prefix)
    model, name = get_model(model, threads, seed=seed)
    train_scores, test_scores = [], []
    tests, preds = None, None
    best_model = None
    best_score = -np.Inf
    print('>', name)
    print('Cross validation:')
    for i, (train_index, test_index) in enumerate(splits):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
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
    print('Average validation metrics:')
    scores_fname = '{}.{}.scores'.format(prefix, name)
    metric_names = (
        'r2_score explained_variance_score mean_absolute_error mean_squared_error'
        .split())
    with open(scores_fname, 'w') as scores_file:
        for m in metric_names:
            try:
                s = getattr(metrics, m)(tests, preds)
                print(' ', score_format(m, s))
                scores_file.write(score_format(m, s, eol='\n'))
            except Exception:
                pass
        scores_file.write('\nModel:\n{}\n\n'.format(model))
    print()
    top_features = top_important_features(best_model, features)
    if top_features is not None:
        fea_fname = '{}.{}.features'.format(prefix, name)
        with open(fea_fname, 'w') as fea_file:
            fea_file.write(sprint_features(top_features))
    score = metrics.r2_score(tests, preds)
    return score
