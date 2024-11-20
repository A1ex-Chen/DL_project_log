def get_cl_meta_df(data_root: str, int_dtype: type=np.int8):
    """df = get_cl_meta_df('./data/')

    This function loads the metadata for cell lines, process it and return
    as a dataframe. The processing includes:
        * change column names to ['data_src', 'site', 'type', 'category'];
        * remove the '-' in cell line names;
        * convert data types for more compact structure;

    Note that if the dataframe is already stored in the processed folder,
    the function simply read from file and return after converting dtypes.

    Args:
        data_root (str): path to the data root folder.
        int_dtype (type): int dtype for storage in RAM.

    Returns:
        pd.DataFrame: processed cell line metadata dataframe.
    """
    df_filename = 'cl_meta_df.pkl'
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)
    else:
        logger.debug('Processing cell line meta dataframe ... ')
        download_files(filenames=CL_METADATA_FILENAME, target_folder=os.
            path.join(data_root, RAW_FOLDER))
        df = pd.read_csv(os.path.join(data_root, RAW_FOLDER,
            CL_METADATA_FILENAME), sep='\t', header=0, index_col=0, usecols
            =['sample_name', 'dataset', 'simplified_tumor_site',
            'simplified_tumor_type', 'sample_category'], dtype=str)
        df.index.names = ['sample']
        df.columns = ['data_src', 'site', 'type', 'category']
        print(df.shape)
        df.index = df.index.str.replace('-', '')
        print(df.shape)
        columns = df.columns
        dict_names = [(i + '_dict.txt') for i in columns]
        for col, dict_name in zip(columns, dict_names):
            df[col] = encode_label_to_int(data_root=data_root, dict_name=
                dict_name, labels=df[col])
        df = df.astype(int)
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        df.to_pickle(df_path)
    df = df.astype(int_dtype)
    return df
