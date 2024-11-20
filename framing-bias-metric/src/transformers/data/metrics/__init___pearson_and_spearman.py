def pearson_and_spearman(preds, labels):
    warnings.warn(DEPRECATION_WARNING, FutureWarning)
    requires_sklearn(pearson_and_spearman)
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {'pearson': pearson_corr, 'spearmanr': spearman_corr, 'corr': (
        pearson_corr + spearman_corr) / 2}
