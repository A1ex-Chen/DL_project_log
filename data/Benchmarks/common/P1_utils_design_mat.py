def design_mat(mod, numerical_covariates, batch_levels):
    design = patsy.dmatrix('~ 0 + C(batch, levels=%s)' % str(batch_levels),
        mod, return_type='dataframe')
    mod = mod.drop(['batch'], axis=1)
    numerical_covariates = list(numerical_covariates)
    sys.stdout.write('found %i batches\n' % design.shape[1])
    other_cols = [c for i, c in enumerate(mod.columns) if i not in
        numerical_covariates]
    factor_matrix = mod[other_cols]
    design = pd.concat((design, factor_matrix), axis=1)
    if numerical_covariates is not None:
        sys.stdout.write('found %i numerical covariates...\n' % len(
            numerical_covariates))
        for i, nC in enumerate(numerical_covariates):
            cname = mod.columns[nC]
            sys.stdout.write('\t{0}\n'.format(cname))
            design[cname] = mod[mod.columns[nC]]
    sys.stdout.write('found %i categorical variables:' % len(other_cols))
    sys.stdout.write('\t' + ', '.join(other_cols) + '\n')
    return design
