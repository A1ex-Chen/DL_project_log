def get_combo_stats_df(data_root: str, grth_scaling: str, int_dtype: type=
    np.int8, float_dtype: type=np.float32):
    """df = get_combo_stats_df('./data/', 'std')

    This function loads the whole drug response file, takes out every single
    drug + cell line combinations, and calculates the statistics including:
        * number of drug response records per combo;
        * average growth per combo;
        * correlation between drug log concentration and growth per combo;
    for all the combinations.

    Note that if the dataframe is already stored in the processed folder,
    the function simply read from file and return after converting dtypes.

    Args:
        data_root (str): path to the data root folder.
        grth_scaling (str): scaling strategy for growth in drug response.
        int_dtype (type): int dtype for storage in RAM.
        float_dtype (float): int dtype for storage in RAM.

    Returns:
        pd.DataFrame: drug cell combination statistics dataframe, each row
            contains the following fields: ['DRUG_ID', 'CELLNAME','NUM_REC',
            'AVG_GRTH', 'CORR'].
    """
    df_filename = 'combo_stats_df(scaling=%s).pkl' % grth_scaling
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)
    else:
        logger.debug(
            'Processing drug + cell combo statics dataframe ... this may take up to 5 minutes.'
            )
        drug_resp_df = get_drug_resp_df(data_root=data_root, grth_scaling=
            grth_scaling, int_dtype=int, float_dtype=float)
        combo_dict = {}
        drug_resp_array = drug_resp_df.values
        for row in drug_resp_array:
            combo = row[1] + '+' + row[2]
            if combo not in combo_dict:
                combo_dict[combo] = [row[1], row[2], (), ()]
            combo_dict[combo][2] += row[3],
            combo_dict[combo][3] += row[4],

        def process_combo(dict_value: list):
            conc_tuple = dict_value[2]
            grth_tuple = dict_value[3]
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    corr = stats.pearsonr(x=conc_tuple, y=grth_tuple)[0]
                except (Warning, ValueError):
                    corr = 0.0
            return [dict_value[0], dict_value[1], len(conc_tuple), np.mean(
                grth_tuple), corr]
        num_cores = multiprocessing.cpu_count()
        combo_stats = Parallel(n_jobs=num_cores)(delayed(process_combo)(v) for
            _, v in combo_dict.items())
        cols = ['DRUG_ID', 'CELLNAME', 'NUM_REC', 'AVG_GRTH', 'CORR']
        df = pd.DataFrame(combo_stats, columns=cols)
        df[['NUM_REC']] = df[['NUM_REC']].astype(int)
        df[['AVG_GRTH', 'CORR']] = df[['AVG_GRTH', 'CORR']].astype(float)
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        df.to_pickle(df_path)
    df[['NUM_REC']] = df[['NUM_REC']].astype(int_dtype)
    df[['AVG_GRTH', 'CORR']] = df[['AVG_GRTH', 'CORR']].astype(float_dtype)
    return df
