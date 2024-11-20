def generalization_feature_selection(data1, data2, measure, cutoff):
    """
    This function uses the Pearson correlation coefficient to select the features that are generalizable
    between data1 and data2.

    Parameters:
    -----------
    data1: 2D numpy array of the first dataset with a size of (n_samples_1, n_features)
    data2: 2D numpy array of the second dataset with a size of (n_samples_2, n_features)
    measure: string. 'pearson' indicates the Pearson correlation coefficient;
        'ccc' indicates the concordance correlation coefficient. Default is 'pearson'.
    cutoff: a positive number for selecting generalizable features. If cutoff < 1, this function selects
        the features with a correlation coefficient >= cutoff. If cutoff >= 1, it must be an
        integer indicating the number of features to be selected based on correlation coefficient.

    Returns:
    --------
    fid: 1-D numpy array containing the indices of selected features.
    """
    cor1 = np.corrcoef(np.transpose(data1))
    cor2 = np.corrcoef(np.transpose(data2))
    num = data1.shape[1]
    cor = []
    if measure == 'pearson':
        for i in range(num):
            cor.append(np.corrcoef(np.vstack((list(cor1[:i, i]) + list(cor1
                [i + 1:, i]), list(cor2[:i, i]) + list(cor2[i + 1:, i]))))[
                0, 1])
    elif measure == 'ccc':
        for i in range(num):
            cor.append(calculate_concordance_correlation_coefficient(np.
                array(list(cor1[:i, i]) + list(cor1[i + 1:, i])), np.array(
                list(cor2[:i, i]) + list(cor2[i + 1:, i]))))
    cor = np.array(cor)
    fid = np.argsort(-cor)[:int(cutoff)]
    return fid
