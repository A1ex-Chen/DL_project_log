def coxen_multi_drug_gene_selection(source_data, target_data,
    drug_response_data, drug_response_col, tumor_col, drug_col,
    prediction_power_measure='lm', num_predictive_gene=100,
    generalization_power_measure='ccc', num_generalizable_gene=50,
    union_of_single_drug_selection=False):
    """
    This function uses the COXEN approach to select genes for predicting the response of multiple drugs.
    It assumes no missing data exist. It works in three modes.
    (1) If union_of_single_drug_selection is True, prediction_power_measure must be either 'pearson' or 'mutual_info'.
    This functions runs coxen_single_drug_gene_selection for every drug with the parameter setting and takes the
    union of the selected genes of every drug as the output. The size of the selected gene set may be larger than
    num_generalizable_gene.
    (2) If union_of_single_drug_selection is False and prediction_power_measure is 'lm', this function uses a
    linear model to fit the response of multiple drugs using the expression of a gene, while the drugs are
    one-hot encoded. The p-value associated with the coefficient of gene expression is used as the prediction
    power measure, according to which num_predictive_gene genes will be selected. Then, among the predictive
    genes, num_generalizable_gene generalizable genes will be  selected.
    (3) If union_of_single_drug_selection is False and prediction_power_measure is 'pearson' or 'mutual_info',
    for each drug this functions ranks the genes according to their power of predicting the
    response of the drug. The union of an equal number of predictive genes for every drug will be generated,
    and its size must be at least num_predictive_gene. Then, num_generalizable_gene generalizable genes
    will be selected.

    Parameters:
    -----------
    source_data: pandas data frame of gene expressions of tumors, for which drug response is known. Its size is
        [n_source_samples, n_features].
    target_data: pandas data frame of gene expressions of tumors, for which drug response needs to be predicted.
        Its size is [n_target_samples, n_features]. source_data and target_data have the same set
        of features and the orders of features must match.
    drug_response_data: pandas data frame of drug response that must include a column of drug response values,
        a column of tumor IDs, and a column of drug IDs.
    drug_response_col: non-negative integer or string. If integer, it is the column index of drug response in
        drug_response_data. If string, it is the column name of drug response.
    tumor_col: non-negative integer or string. If integer, it is the column index of tumor IDs in drug_response_data.
        If string, it is the column name of tumor IDs.
    drug_col: non-negative integer or string. If integer, it is the column index of drugs in drug_response_data.
        If string, it is the column name of drugs.
    prediction_power_measure: string. 'pearson' uses the absolute value of Pearson correlation coefficient to
        measure prediction power of a gene; 'mutual_info' uses the mutual information to measure prediction power
        of a gene; 'lm' uses the linear regression model to select predictive genes for multiple drugs. Default is 'lm'.
    num_predictive_gene: positive integer indicating the number of predictive genes to be selected.
    generalization_power_measure: string. 'pearson' indicates the Pearson correlation coefficient;
        'ccc' indicates the concordance correlation coefficient. Default is 'ccc'.
    num_generalizable_gene: positive integer indicating the number of generalizable genes to be selected.
    union_of_single_drug_selection: boolean, indicating whether the final gene set should be the union of genes
        selected for every drug.

    Returns:
    --------
    indices: 1-D numpy array containing the indices of selected genes.
    """
    if isinstance(drug_response_col, str):
        drug_response_col = np.where(drug_response_data.columns ==
            drug_response_col)[0][0]
    if isinstance(tumor_col, str):
        tumor_col = np.where(drug_response_data.columns == tumor_col)[0][0]
    if isinstance(drug_col, str):
        drug_col = np.where(drug_response_data.columns == drug_col)[0][0]
    drug_response_data = drug_response_data.copy()
    drug_response_data = drug_response_data.iloc[np.where(np.isin(
        drug_response_data.iloc[:, tumor_col], source_data.index))[0], :]
    drugs = np.unique(drug_response_data.iloc[:, drug_col])
    source_data = source_data.copy()
    source_data = source_data.iloc[np.where(np.isin(source_data.index,
        drug_response_data.iloc[:, tumor_col]))[0], :]
    source_std_id = select_features_by_variation(source_data,
        variation_measure='std', threshold=1e-08)
    target_std_id = select_features_by_variation(target_data,
        variation_measure='std', threshold=1e-08)
    std_id = np.sort(np.intersect1d(source_std_id, target_std_id))
    source_data = source_data.iloc[:, std_id]
    target_data = target_data.copy()
    target_data = target_data.iloc[:, std_id]
    num_predictive_gene = int(min(num_predictive_gene, source_data.shape[1]))
    if union_of_single_drug_selection:
        if (prediction_power_measure != 'pearson' and 
            prediction_power_measure != 'mutual_info'):
            print(
                'pearson or mutual_info must be used as prediction_power_measure for taking the union of selected genes of every drugs'
                )
            sys.exit(1)
        gid1 = np.array([]).astype(np.int64)
        for d in drugs:
            idd = np.where(drug_response_data.iloc[:, drug_col] == d)[0]
            response_d = drug_response_data.iloc[idd, :]
            gid2 = coxen_single_drug_gene_selection(source_data,
                target_data, response_d, drug_response_col, tumor_col,
                prediction_power_measure, num_predictive_gene,
                generalization_power_measure, num_generalizable_gene)
            gid1 = np.union1d(gid1, gid2)
        return np.sort(std_id[gid1])
    if prediction_power_measure == 'lm':
        pvalue = np.empty((source_data.shape[1], 1))
        pvalue.fill(np.nan)
        drug_m = np.identity(len(drugs))
        drug_m = pd.DataFrame(drug_m, index=drugs)
        drug_sample = drug_m.loc[drug_response_data.iloc[:, drug_col], :
            ].values
        for i in range(source_data.shape[1]):
            ge_sample = source_data.iloc[:, i].loc[drug_response_data.iloc[
                :, tumor_col]].values
            sample = np.hstack((np.reshape(ge_sample, (len(ge_sample), 1)),
                drug_sample))
            sample = sm.add_constant(sample)
            mod = sm.OLS(drug_response_data.iloc[:, drug_response_col].
                values, sample)
            try:
                res = mod.fit()
                pvalue[i, 0] = res.pvalues[1]
            except ValueError:
                pvalue[i, 0] = 1
        gid1 = np.argsort(pvalue[:, 0])[:num_predictive_gene]
    elif prediction_power_measure == 'pearson' or prediction_power_measure == 'mutual_info':
        gene_rank = np.empty((len(drugs), source_data.shape[1]))
        gene_rank.fill(np.nan)
        gene_rank = pd.DataFrame(gene_rank, index=drugs)
        for d in range(len(drugs)):
            idd = np.where(drug_response_data.iloc[:, drug_col] == drugs[d])[0]
            response_d = drug_response_data.iloc[idd, :]
            temp_rank = coxen_single_drug_gene_selection(source_data,
                target_data, response_d, drug_response_col, tumor_col,
                prediction_power_measure, num_predictive_gene=None,
                generalization_power_measure=None, num_generalizable_gene=
                None, multi_drug_mode=True)
            gene_rank.iloc[d, :len(temp_rank)] = temp_rank
        for i in range(int(np.ceil(num_predictive_gene / len(drugs))), 
            source_data.shape[1] + 1):
            gid1 = np.unique(np.reshape(gene_rank.iloc[:, :i].values, (1, 
                gene_rank.shape[0] * i))[0, :])
            gid1 = gid1[np.where(np.invert(np.isnan(gid1)))[0]]
            if len(gid1) >= num_predictive_gene:
                break
        gid1 = gid1.astype(np.int64)
    source_data = source_data.iloc[:, gid1]
    target_data = target_data.iloc[:, gid1]
    num_generalizable_gene = int(min(num_generalizable_gene, len(gid1)))
    gid2 = generalization_feature_selection(source_data.values, target_data
        .values, generalization_power_measure, num_generalizable_gene)
    indices = std_id[gid1[gid2]]
    return np.sort(indices)
