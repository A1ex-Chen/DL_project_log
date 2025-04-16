def get_drug_fgpt_df(data_root: str, int_dtype: type=np.int8):
    """df = get_drug_fgpt_df('./data/')

    This function loads two drug fingerprint files, join them as one
    dataframe, convert them to int_dtype and return.

    Note that if the dataframe is already stored in the processed folder,
    the function simply read from file and return after converting dtypes.

    Args:
        data_root (str): path to the data root folder.
        int_dtype (type): int dtype for storage in RAM.

    Returns:
        pd.DataFrame: processed drug fingerprint dataframe.
    """
    df_filename = 'drug_fgpt_df.pkl'
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)
    else:
        logger.debug('Processing drug fingerprint dataframe ... ')
        download_files(filenames=[ECFP_FILENAME, PFP_FILENAME],
            target_folder=os.path.join(data_root, RAW_FOLDER))
        ecfp_df = pd.read_csv(os.path.join(data_root, RAW_FOLDER,
            ECFP_FILENAME), sep='\t', header=None, index_col=0, skiprows=[0])
        pfp_df = pd.read_csv(os.path.join(data_root, RAW_FOLDER,
            PFP_FILENAME), sep='\t', header=None, index_col=0, skiprows=[0])
        df = pd.concat([ecfp_df, pfp_df], axis=1, join='inner')
        df = df.astype(int)
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        df.to_pickle(df_path)
    df = df.astype(int_dtype)
    return df
