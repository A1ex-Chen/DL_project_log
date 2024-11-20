def get_drug_feature_df(data_root: str, drug_feature_usage: str,
    dscptr_scaling: str, dscptr_nan_thresh: float, int_dtype: type=np.int8,
    float_dtype: type=np.float32):
    """df = get_drug_feature_df('./data/', 'both', 'std', 0.0)

    This function utilizes get_drug_fgpt_df and get_drug_dscptr_df. If the
    feature usage is 'both', it will loads fingerprint and descriptors,
    join them and return. Otherwise, if feature usage is set to
    'fingerprint' or 'descriptor', the function returns the corresponding
    dataframe.

    Args:
        data_root (str): path to the data root folder.
        drug_feature_usage (str): feature usage indicator. Choose between
            'both', 'fingerprint', and 'descriptor'.
        dscptr_scaling (str): scaling strategy for all descriptor features.
        dscptr_nan_thresh (float): threshold ratio of NaN values.
        int_dtype (type): int dtype for storage in RAM.
        float_dtype (float): int dtype for storage in RAM.

    Returns:
        pd.DataFrame: processed drug feature dataframe.
    """
    if drug_feature_usage == 'both':
        return pd.concat([get_drug_fgpt_df(data_root=data_root, int_dtype=
            int_dtype), get_drug_dscptr_df(data_root=data_root,
            dscptr_scaling=dscptr_scaling, dscptr_nan_thresh=
            dscptr_nan_thresh, float_dtype=float_dtype)], axis=1, join='inner')
    elif drug_feature_usage == 'fingerprint':
        return get_drug_fgpt_df(data_root=data_root, int_dtype=int_dtype)
    elif drug_feature_usage == 'descriptor':
        return get_drug_dscptr_df(data_root=data_root, dscptr_scaling=
            dscptr_scaling, dscptr_nan_thresh=dscptr_nan_thresh,
            float_dtype=float_dtype)
    else:
        logger.error(
            "Drug feature must be one of 'fingerprint', 'descriptor', or 'both'."
            , exc_info=True)
        raise ValueError('Undefined drug feature %s.' % drug_feature_usage)
