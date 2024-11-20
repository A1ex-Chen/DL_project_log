def get_drug_dscptr_df(data_root: str, dscptr_scaling: str,
    dscptr_nan_thresh: float, float_dtype: type=np.float32):
    """df = get_drug_dscptr_df('./data/', 'std', 0.0)

    This function loads the drug descriptor file, process it and return
    as a dataframe. The processing includes:
        * removing columns (features) and rows (drugs) that have exceeding
            ratio of NaN values comparing to nan_thresh;
        * scaling all the descriptor features accordingly;
        * convert data types for more compact structure;

    Note that if the dataframe is already stored in the processed folder,
    the function simply read from file and return after converting dtypes.

    Args:
        data_root (str): path to the data root folder.
        dscptr_scaling (str): scaling strategy for all descriptor features.
        dscptr_nan_thresh (float): threshold ratio of NaN values.
        float_dtype (float): float dtype for storage in RAM.

    Returns:
        pd.DataFrame: processed drug descriptor dataframe.
    """
    df_filename = 'drug_dscptr_df(scaling=%s, nan_thresh=%.2f).pkl' % (
        dscptr_scaling, dscptr_nan_thresh)
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)
    else:
        logger.debug('Processing drug descriptor dataframe ... ')
        download_files(filenames=DSCPTR_FILENAME, target_folder=os.path.
            join(data_root, RAW_FOLDER))
        df = pd.read_csv(os.path.join(data_root, RAW_FOLDER,
            DSCPTR_FILENAME), sep='\t', header=0, index_col=0, na_values='na')
        valid_thresh = 1.0 - dscptr_nan_thresh
        df.dropna(axis=1, inplace=True, thresh=int(df.shape[0] * valid_thresh))
        df.dropna(axis=0, inplace=True, thresh=int(df.shape[1] * valid_thresh))
        df.fillna(df.mean(), inplace=True)
        df = scale_dataframe(df, dscptr_scaling)
        df = df.astype(float)
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        df.to_pickle(df_path)
    df = df.astype(float_dtype)
    return df
