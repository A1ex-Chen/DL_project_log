def get_drug_target_df(data_root: str, int_dtype: type=np.int8):
    """df = get_drug_target_df('./data/')

    This function the drug property dataframe, process it and return the
    drug target families dataframe. The processing includes:
        * removing all columns but 'TARGET';
        * drop drugs/rows that are not in the TGT_FAMS list;
        * encode target families into integer labels;
        * convert data types for more compact structure;

    Args:
        data_root (str): path to the data root folder.
        int_dtype (type): int dtype for storage in RAM.

    Returns:
        pd.DataFrame: drug target families dataframe.
    """
    df = get_drug_prop_df(data_root=data_root)[['TARGET']]
    df = df[df['TARGET'].isin(TGT_FAMS)][['TARGET']]
    df['TARGET'] = encode_label_to_int(data_root=data_root, dict_name=
        'drug_target_dict.txt', labels=df['TARGET'])
    return df.astype(int_dtype)
