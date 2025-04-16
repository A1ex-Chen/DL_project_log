def get_drug_prop_df(data_root: str):
    """df = get_drug_prop_df('./data/')

    This function loads the drug property file and returns a dataframe with
    only weighted QED and target families as columns (['QED', 'TARGET']).

    Note that if the dataframe is already stored in the processed folder,
    the function simply read from file and return after converting dtypes.

    Args:
        data_root (str): path to the data root folder.

    Returns:
        pd.DataFrame: drug property dataframe with target families and QED.
    """
    df_filename = 'drug_prop_df.pkl'
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)
    else:
        logger.debug('Processing drug targets dataframe ... ')
        raw_file_path = os.path.join(data_root, RAW_FOLDER, DRUG_PROP_FILENAME)
        download_files(filenames=DRUG_PROP_FILENAME, target_folder=os.path.
            join(data_root, RAW_FOLDER))
        df = pd.read_csv(raw_file_path, sep='\t', header=0, index_col=0,
            usecols=['anl_cpd_id', 'qed_weighted', 'target_families'])
        df.index.names = ['DRUG_ID']
        df.columns = ['QED', 'TARGET']
        df[['QED']] = df[['QED']].astype(float)
        df[['TARGET']] = df[['TARGET']].astype(str)
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        df.to_pickle(df_path)
    return df
