def get_rna_seq_df(data_root: str, rnaseq_feature_usage: str,
    rnaseq_scaling: str, float_dtype: type=np.float32):
    """df = get_rna_seq_df('./data/', 'source_scale', 'std')

    This function loads the RNA sequence file, process it and return
    as a dataframe. The processing includes:
        * remove the '-' in cell line names;
        * remove duplicate indices;
        * scaling all the sequence features accordingly;
        * convert data types for more compact structure;

    Note that if the dataframe is already stored in the processed folder,
    the function simply read from file and return after converting dtypes.

    Args:
        data_root (str): path to the data root folder.
        rnaseq_feature_usage (str): feature usage indicator, Choose between
            'source_scale' and 'combat'.
        rnaseq_scaling (str): scaling strategy for RNA sequence.
        float_dtype (float): int dtype for storage in RAM.

    Returns:
        pd.DataFrame: processed RNA sequence dataframe.
    """
    df_filename = 'rnaseq_df(%s, scaling=%s).pkl' % (rnaseq_feature_usage,
        rnaseq_scaling)
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)
    else:
        logger.debug('Processing RNA sequence dataframe ... ')
        if rnaseq_feature_usage == 'source_scale':
            raw_data_filename = RNASEQ_SOURCE_SCALE_FILENAME
        elif rnaseq_feature_usage == 'combat':
            raw_data_filename = RNASEQ_COMBAT_FILENAME
        else:
            logger.error('Unknown RNA feature %s.' % rnaseq_feature_usage,
                exc_info=True)
            raise ValueError(
                "RNA feature usage must be one of 'source_scale' or 'combat'.")
        download_files(filenames=raw_data_filename, target_folder=os.path.
            join(data_root, RAW_FOLDER))
        df = pd.read_csv(os.path.join(data_root, RAW_FOLDER,
            raw_data_filename), sep='\t', header=0, index_col=0)
        df.index = df.index.str.replace('-', '')
        print(df.shape)
        df = df[~df.index.duplicated(keep='first')]
        print(df.shape)
        df = scale_dataframe(df, rnaseq_scaling)
        df = df.astype(float)
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        df.to_pickle(df_path)
    df = df.astype(float_dtype)
    return df
