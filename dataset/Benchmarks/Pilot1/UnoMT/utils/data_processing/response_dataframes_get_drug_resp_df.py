def get_drug_resp_df(data_root: str, grth_scaling: str, int_dtype: type=np.
    int8, float_dtype: type=np.float32):
    """df = get_drug_resp_df('./data/', 'std')

    This function loads the whole drug response file, process it and return
    as a dataframe. The processing includes:
        * remove the '-' in cell line names;
        * encode str format data sources into integer;
        * scaling the growth accordingly;
        * convert data types for more compact structure;

    Note that if the dataframe is already stored in the processed folder,
    the function simply read from file and return after converting dtypes.

    Args:
        data_root (str): path to the data root folder.
        grth_scaling (str): scaling strategy for growth in drug response.
        int_dtype (type): int dtype for storage in RAM.
        float_dtype (float): int dtype for storage in RAM.

    Returns:
        pd.DataFrame: processed drug response dataframe.
    """
    df_filename = 'drug_resp_df(scaling=%s).pkl' % grth_scaling
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)
    else:
        logger.debug('Processing drug response dataframe ... ')
        download_files(filenames=DRUG_RESP_FILENAME, target_folder=os.path.
            join(data_root, RAW_FOLDER))
        df = pd.read_csv(os.path.join(data_root, RAW_FOLDER,
            DRUG_RESP_FILENAME), sep='\t', header=0, index_col=None,
            usecols=[0, 1, 2, 4, 6])
        df['CELLNAME'] = df['CELLNAME'].str.replace('-', '')
        df['SOURCE'] = encode_label_to_int(data_root=data_root, dict_name=
            'data_src_dict.txt', labels=df['SOURCE'].tolist())
        df['GROWTH'] = scale_dataframe(df['GROWTH'], grth_scaling)
        df[['SOURCE']] = df[['SOURCE']].astype(int)
        df[['LOG_CONCENTRATION', 'GROWTH']] = df[['LOG_CONCENTRATION',
            'GROWTH']].astype(float)
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        df.to_pickle(df_path)
    df[['SOURCE']] = df[['SOURCE']].astype(int_dtype)
    df[['LOG_CONCENTRATION', 'GROWTH']] = df[['LOG_CONCENTRATION', 'GROWTH']
        ].astype(float_dtype)
    return df
