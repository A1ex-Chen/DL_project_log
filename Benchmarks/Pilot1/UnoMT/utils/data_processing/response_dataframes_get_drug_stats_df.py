def get_drug_stats_df(data_root: str, grth_scaling: str, int_dtype: type=np
    .int16, float_dtype: type=np.float32):
    """df = get_drug_stats_df('./data/', 'std')

    This function loads the combination statistics file, iterates through
    all the drugs, and calculated the statistics including:
        * number of cell lines tested per drug;
        * number of drug response records per drug;
        * average of growth per drug;
        * average correlation of dose and growth per drug;
    for all the drugs.

    Note that if the dataframe is already stored in the processed folder,
    the function simply read from file and return after converting dtypes.

    Args:
        data_root (str): path to the data root folder.
        grth_scaling (str): scaling strategy for growth in drug response.
        int_dtype (type): int dtype for storage in RAM.
        float_dtype (float): int dtype for storage in RAM.

    Returns:
        pd.DataFrame: drug cell combination statistics dataframe, each row
            contains the following fields: ['DRUG_ID', 'NUM_CL', 'NUM_REC',
            'AVG_GRTH', 'AVG_CORR']
    """
    if int_dtype == np.int8:
        logger.warning('Integer length too smaller for drug statistics.')
    df_filename = 'drug_stats_df(scaling=%s).pkl' % grth_scaling
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)
    else:
        logger.debug('Processing drug statics dataframe ... ')
        combo_stats_df = get_combo_stats_df(data_root=data_root,
            grth_scaling=grth_scaling, int_dtype=int, float_dtype=float)
        drug_dict = {}
        combo_stats_array = combo_stats_df.values
        for row in combo_stats_array:
            drug = row[0]
            if drug not in drug_dict:
                drug_dict[drug] = [0, (), (), ()]
            drug_dict[drug][0] += 1
            drug_dict[drug][1] += row[2],
            drug_dict[drug][2] += row[3],
            drug_dict[drug][3] += row[4],

        def process_drug(drug: str, dict_value: list):
            num_cl = dict_value[0]
            records_tuple = dict_value[1]
            assert num_cl == len(records_tuple)
            grth_tuple = dict_value[2]
            corr_tuple = dict_value[3]
            num_rec = np.sum(records_tuple)
            avg_grth = np.average(a=grth_tuple, weights=records_tuple)
            avg_corr = np.average(a=corr_tuple, weights=records_tuple)
            return [drug, num_cl, num_rec, avg_grth, avg_corr]
        num_cores = multiprocessing.cpu_count()
        drug_stats = Parallel(n_jobs=num_cores)(delayed(process_drug)(k, v) for
            k, v in drug_dict.items())
        cols = ['DRUG_ID', 'NUM_CL', 'NUM_REC', 'AVG_GRTH', 'AVG_CORR']
        df = pd.DataFrame(drug_stats, columns=cols)
        df[['NUM_CL', 'NUM_REC']] = df[['NUM_CL', 'NUM_REC']].astype(int)
        df[['AVG_GRTH', 'AVG_CORR']] = df[['AVG_GRTH', 'AVG_CORR']].astype(
            float)
        df.set_index('DRUG_ID', inplace=True)
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        df.to_pickle(df_path)
    df[['NUM_CL', 'NUM_REC']] = df[['NUM_CL', 'NUM_REC']].astype(int_dtype)
    df[['AVG_GRTH', 'AVG_CORR']] = df[['AVG_GRTH', 'AVG_CORR']].astype(
        float_dtype)
    return df
