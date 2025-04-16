def coxen_single_drug_gene_selection(source_data, target_data,
    drug_response_data, drug_response_col, tumor_col,
    prediction_power_measure='pearson', num_predictive_gene=100,
    generalization_power_measure='ccc', num_generalizable_gene=50,
    multi_drug_mode=False):
    """
    This function selects genes for drug response prediction using the COXEN approach. The COXEN approach is
    designed for selecting genes to predict the response of tumor cells to a specific drug. This function
    assumes no missing data exist.

    Parameters:
    -----------
    source_data: pandas data frame of gene expressions of tumors, for which drug response is known. Its size is
        [n_source_samples, n_features].
    target_data: pandas data frame of gene expressions of tumors, for which drug response needs to be predicted.
        Its size is [n_target_samples, n_features]. source_data and target_data have the same set
        of features and the orders of features must match.
    drug_response_data: pandas data frame of drug response values for a drug. It must include a column of drug
        response values and a column of tumor IDs.
    drug_response_col: non-negative integer or string. If integer, it is the column index of drug response in
        drug_response_data. If string, it is the column name of drug response.
    tumor_col: non-negative integer or string. If integer, it is the column index of tumor IDs in drug_response_data.
        If string, it is the column name of tumor IDs.
    prediction_power_measure: string. 'pearson' uses the absolute value of Pearson correlation coefficient to
        measure prediction power of gene; 'mutual_info' uses the mutual information to measure prediction power
        of gene. Default is 'pearson'.
    num_predictive_gene: positive integer indicating the number of predictive genes to be selected.
    generalization_power_measure: string. 'pearson' indicates the Pearson correlation coefficient;
        'ccc' indicates the concordance correlation coefficient. Default is 'ccc'.
    num_generalizable_gene: positive integer indicating the number of generalizable genes to be selected.
    multi_drug_mode: boolean, indicating whether the function runs as an auxiliary function of COXEN
        gene selection for multiple drugs. Default is False.

    Returns:
    --------
    indices: 1-D numpy array containing the indices of selected genes, if multi_drug_mode is False;
    1-D numpy array of indices of sorting all genes according to their prediction power, if multi_drug_mode is True.
    """
    if isinstance(drug_response_col, str):
        drug_response_col = np.where(drug_response_data.columns ==
            drug_response_col)[0][0]
    if isinstance(tumor_col, str):
        tumor_col = np.where(drug_response_data.columns == tumor_col)[0][0]
    drug_response_data = drug_response_data.copy()
    drug_response_data = drug_response_data.iloc[np.where(np.isin(
        drug_response_data.iloc[:, tumor_col], source_data.index))[0], :]
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
    batchSize = 1000
    numBatch = int(np.ceil(source_data.shape[1] / batchSize))
    prediction_power = np.empty((source_data.shape[1], 1))
    prediction_power.fill(np.nan)
    for i in range(numBatch):
        startIndex = i * batchSize
        endIndex = min((i + 1) * batchSize, source_data.shape[1])
        if prediction_power_measure == 'pearson':
            cor_i = np.corrcoef(np.vstack((np.transpose(source_data.iloc[:,
                startIndex:endIndex].loc[drug_response_data.iloc[:,
                tumor_col], :].values), np.reshape(drug_response_data.iloc[
                :, drug_response_col].values, (1, drug_response_data.shape[
                0])))))
            prediction_power[startIndex:endIndex, 0] = abs(cor_i[:-1, -1])
        if prediction_power_measure == 'mutual_info':
            mi = mutual_info_regression(X=source_data.iloc[:, startIndex:
                endIndex].loc[drug_response_data.iloc[:, tumor_col], :].
                values, y=drug_response_data.iloc[:, drug_response_col].values)
            prediction_power[startIndex:endIndex, 0] = mi
    if multi_drug_mode:
        indices = np.argsort(-prediction_power[:, 0])
        return std_id[indices]
    num_predictive_gene = int(min(num_predictive_gene, source_data.shape[1]))
    gid1 = np.argsort(-prediction_power[:, 0])[:num_predictive_gene]
    source_data = source_data.iloc[:, gid1]
    target_data = target_data.iloc[:, gid1]
    num_generalizable_gene = int(min(num_generalizable_gene, len(gid1)))
    gid2 = generalization_feature_selection(source_data.values, target_data
        .values, generalization_power_measure, num_generalizable_gene)
    indices = std_id[gid1[gid2]]
    return np.sort(indices)
