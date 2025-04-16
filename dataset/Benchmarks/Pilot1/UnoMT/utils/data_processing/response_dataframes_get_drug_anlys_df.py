def get_drug_anlys_df(data_root: str):
    """df = get_drug_anlys_df('./data/')

    This function will load the drug statistics dataframe and go on and
    classify all the drugs into 4 different categories:
        * high growth, high correlation
        * high growth, low correlation
        * low growth, high correlation
        * low growth, low correlation
    Using the median value of growth and correlation. The results will be
    returned as a dataframe with drug ID as index.

    Note that if the dataframe is already stored in the processed folder,
    the function simply read from file and return after converting dtypes.

    Args:
        data_root (str): path to the data root folder.

    Returns:
        pd.DataFrame: drug classes with growth and correlation, each row
            contains the following fields: ['HIGH_GROWTH', 'HIGH_CORR'],
            which are boolean features.
    """
    df_filename = 'drug_anlys_df.pkl'
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)
    if os.path.exists(df_path):
        return pd.read_pickle(df_path)
    else:
        logger.debug('Processing drug analysis dataframe ... ')
        drug_stats_df = get_drug_stats_df(data_root=data_root, grth_scaling
            ='none', int_dtype=int, float_dtype=float)
        drugs = drug_stats_df.index
        avg_grth = drug_stats_df['AVG_GRTH'].values
        avg_corr = drug_stats_df['AVG_CORR'].values
        high_grth = avg_grth > np.median(avg_grth)
        high_corr = avg_corr > np.median(avg_corr)
        drug_analysis_array = np.array([drugs, high_grth, high_corr]
            ).transpose()
        df = pd.DataFrame(drug_analysis_array, columns=['DRUG_ID',
            'HIGH_GROWTH', 'HIGH_CORR'])
        df.set_index('DRUG_ID', inplace=True)
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        df.to_pickle(df_path)
        return df
