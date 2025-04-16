def load_dose_response(path, seed, dtype, min_logconc=-5.0, max_logconc=-
    5.0, subsample=None):
    """Load cell line response to different drug compounds, sub-select response for a specific
        drug log concentration range and return a pandas dataframe.

    Parameters
    ----------
    path: string
        path to 'NCI60_dose_response_with_missing_z5_avg.csv'
    seed: integer
        seed for random generation
    dtype: numpy type
        precision (data type) for reading float values
    min_logconc : -3, -4, -5, -6, -7, optional (default -5)
        min log concentration of drug to return cell line growth
    max_logconc : -3, -4, -5, -6, -7, optional (default -5)
        max log concentration of drug to return cell line growth
    subsample: None, 'naive_balancing' (default None)
        subsampling strategy to use to balance the data based on growth
    """
    df = pd.read_csv(path, sep=',', engine='c', na_values=['na', '-', ''],
        dtype={'NSC': object, 'CELLNAME': str, 'LOG_CONCENTRATION': dtype,
        'GROWTH': dtype})
    df = df[(df['LOG_CONCENTRATION'] >= min_logconc) & (df[
        'LOG_CONCENTRATION'] <= max_logconc)]
    df = df[['NSC', 'CELLNAME', 'GROWTH', 'LOG_CONCENTRATION']]
    if subsample and subsample == 'naive_balancing':
        df1 = df[df['GROWTH'] <= 0]
        df2 = df[(df['GROWTH'] > 0) & (df['GROWTH'] < 50)].sample(frac=0.7,
            random_state=seed)
        df3 = df[(df['GROWTH'] >= 50) & (df['GROWTH'] <= 100)].sample(frac=
            0.18, random_state=seed)
        df4 = df[df['GROWTH'] > 100].sample(frac=0.01, random_state=seed)
        df = pd.concat([df1, df2, df3, df4])
    df = df.set_index(['NSC'])
    return df
