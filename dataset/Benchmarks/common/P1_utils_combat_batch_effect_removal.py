def combat_batch_effect_removal(data, batch_labels, model=None,
    numerical_covariates=None):
    """
    This function corrects for batch effect in data.

    Parameters:
    -----------
    data: pandas data frame of numeric values, with a size of (n_features, n_samples)
    batch_labels: pandas series, with a length of n_samples. It should provide the batch labels of samples.
        Its indices are the same as the column names (sample names) in "data".
    model: an object of patsy.design_info.DesignMatrix. It is a design matrix describing the covariate
        information on the samples that could cause batch effects. If not provided, this function
        will attempt to coarsely correct just based on the information provided in "batch".
    numerical_covariates: a list of the names of covariates in "model" that are numerical rather than
        categorical.

    Returns:
    --------
    corrected : pandas data frame of numeric values, with a size of (n_features, n_samples). It is
        the data with batch effects corrected.
    """
    if isinstance(numerical_covariates, str):
        numerical_covariates = [numerical_covariates]
    if numerical_covariates is None:
        numerical_covariates = []
    if model is not None and isinstance(model, pd.DataFrame):
        model['batch'] = list(batch_labels)
    else:
        model = pd.DataFrame({'batch': batch_labels})
    batch_items = model.groupby('batch').groups.items()
    batch_levels = [k for k, v in batch_items]
    batch_info = [v for k, v in batch_items]
    n_batch = len(batch_info)
    n_batches = np.array([len(v) for v in batch_info])
    n_array = float(sum(n_batches))
    drop_cols = [cname for cname, inter in (model == 1).all().iteritems() if
        inter == True]
    drop_idxs = [list(model.columns).index(cdrop) for cdrop in drop_cols]
    model = model[[c for c in model.columns if c not in drop_cols]]
    numerical_covariates = [(list(model.columns).index(c) if isinstance(c,
        str) else c) for c in numerical_covariates if c not in drop_cols]
    design = design_mat(model, numerical_covariates, batch_levels)
    sys.stdout.write('Standardizing Data across genes.\n')
    B_hat = np.dot(np.dot(la.inv(np.dot(design.T, design)), design.T), data.T)
    grand_mean = np.dot((n_batches / n_array).T, B_hat[:n_batch, :])
    var_pooled = np.dot((data - np.dot(design, B_hat).T) ** 2, np.ones((int
        (n_array), 1)) / int(n_array))
    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones
        ((1, int(n_array))))
    tmp = np.array(design.copy())
    tmp[:, :n_batch] = 0
    stand_mean += np.dot(tmp, B_hat).T
    s_data = (data - stand_mean) / np.dot(np.sqrt(var_pooled), np.ones((1,
        int(n_array))))
    sys.stdout.write('Fitting L/S model and finding priors\n')
    batch_design = design[design.columns[:n_batch]]
    gamma_hat = np.dot(np.dot(la.inv(np.dot(batch_design.T, batch_design)),
        batch_design.T), s_data.T)
    delta_hat = []
    for i, batch_idxs in enumerate(batch_info):
        delta_hat.append(s_data[batch_idxs].var(axis=1))
    gamma_bar = gamma_hat.mean(axis=1)
    t2 = gamma_hat.var(axis=1)
    a_prior = list(map(aprior, delta_hat))
    b_prior = list(map(bprior, delta_hat))
    sys.stdout.write('Finding parametric adjustments\n')
    gamma_star, delta_star = [], []
    for i, batch_idxs in enumerate(batch_info):
        temp = it_sol(s_data[batch_idxs], gamma_hat[i], delta_hat[i],
            gamma_bar[i], t2[i], a_prior[i], b_prior[i])
        gamma_star.append(temp[0])
        delta_star.append(temp[1])
    sys.stdout.write('Adjusting data\n')
    bayesdata = s_data
    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)
    for j, batch_idxs in enumerate(batch_info):
        dsq = np.sqrt(delta_star[j, :])
        dsq = dsq.reshape((len(dsq), 1))
        denom = np.dot(dsq, np.ones((1, n_batches[j])))
        numer = np.array(bayesdata[batch_idxs] - np.dot(batch_design.loc[
            batch_idxs], gamma_star).T)
        bayesdata[batch_idxs] = numer / denom
    vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
    bayesdata = bayesdata * np.dot(vpsq, np.ones((1, int(n_array)))
        ) + stand_mean
    return bayesdata
