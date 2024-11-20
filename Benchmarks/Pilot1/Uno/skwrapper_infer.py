def infer(model, df, ycol='0', ignore_categoricals=False, classify=False,
    prefix=''):
    if type(model) == str:
        with open(model, 'rb') as f:
            model = pickle.load(f)
    cat_cols = df.select_dtypes(['object']).columns
    if ignore_categoricals:
        df[cat_cols] = 0
    else:
        df[cat_cols] = df[cat_cols].apply(lambda x: x.astype('category').
            cat.codes)
    if ycol.isdigit():
        ycol = df.columns[int(ycol)]
    y = df.loc[:, ycol].values
    x = df.drop(ycol, axis=1).values
    y_pred = model.predict(x)
    if classify:
        metric_names = (
            'accuracy_score matthews_corrcoef f1_score precision_score recall_score log_loss'
            .split())
    else:
        metric_names = (
            'r2_score explained_variance_score mean_absolute_error mean_squared_error'
            .split())
    scores = {}
    print('Average test metrics:')
    scores_fname = '{}.test.scores'.format(prefix)
    with open(scores_fname, 'w') as scores_file:
        for m in metric_names:
            try:
                s = getattr(metrics, m)(y, y_pred)
                scores[m] = s
                print(' ', score_format(m, s))
                scores_file.write(score_format(m, s, eol='\n'))
            except Exception:
                pass
        scores_file.write('\nModel:\n{}\n\n'.format(model))
    print()
    return scores
