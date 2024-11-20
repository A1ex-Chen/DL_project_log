def get_drug_qed_df(data_root: str, qed_scaling: str, float_dtype: type=np.
    float32):
    """df = get_drug_qed_df('./data/', 'none')


    This function the drug property dataframe, process it and return the
    drug weighted QED dataframe. The processing includes:
        * removing all columns but 'QED';
        * drop drugs/rows that have NaN as weighted QED;
        * scaling the QED accordingly;
        * convert data types for more compact structure;

    Args:
        data_root (str): path to the data root folder.
        qed_scaling (str): scaling strategy for weighted QED.
        float_dtype (float): float dtype for storage in RAM.

    Returns:
        pd.DataFrame: drug weighted QED dataframe.
    """
    df = get_drug_prop_df(data_root=data_root)[['QED']]
    df.dropna(axis=0, inplace=True)
    df = scale_dataframe(df, qed_scaling)
    return df.astype(float_dtype)
