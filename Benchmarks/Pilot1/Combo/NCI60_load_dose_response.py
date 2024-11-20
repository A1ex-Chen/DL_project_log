def load_dose_response(min_logconc=-4.0, max_logconc=-4.0, subsample=None,
    fraction=False):
    """Load cell line response to different drug compounds, sub-select response for a specific
        drug log concentration range and return a pandas dataframe.

    Parameters
    ----------
    min_logconc : -3, -4, -5, -6, -7, optional (default -4)
        min log concentration of drug to return cell line growth
    max_logconc : -3, -4, -5, -6, -7, optional (default -4)
        max log concentration of drug to return cell line growth
    subsample: None, 'naive_balancing' (default None)
        subsampling strategy to use to balance the data based on growth
    fraction: bool (default False)
        divide growth percentage by 100
    """
    path = get_file(P1B3_URL + 'NCI60_dose_response_with_missing_z5_avg.csv')
    df = global_cache.get(path)
    if df is None:
        df = pd.read_csv(path, sep=',', engine='c', na_values=['na', '-',
            ''], dtype={'NSC': object, 'CELLNAME': str, 'LOG_CONCENTRATION':
            np.float32, 'GROWTH': np.float32})
        global_cache[path] = df
    df = df[(df['LOG_CONCENTRATION'] >= min_logconc) & (df[
        'LOG_CONCENTRATION'] <= max_logconc)]
    df = df[['NSC', 'CELLNAME', 'GROWTH', 'LOG_CONCENTRATION']]
    if subsample and subsample == 'naive_balancing':
        df1 = df[df['GROWTH'] <= 0]
        df2 = df[(df['GROWTH'] > 0) & (df['GROWTH'] < 50)].sample(frac=0.7,
            random_state=SEED)
        df3 = df[(df['GROWTH'] >= 50) & (df['GROWTH'] <= 100)].sample(frac=
            0.18, random_state=SEED)
        df4 = df[df['GROWTH'] > 100].sample(frac=0.01, random_state=SEED)
        df = pd.concat([df1, df2, df3, df4])
    if fraction:
        df['GROWTH'] /= 100
    df = df.set_index(['NSC'])
    return df
