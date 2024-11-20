def load_drug_autoencoded(path, dtype, ncols=None, scaling='std'):
    """Load drug latent representation from autoencoder, sub-select
    columns of drugs randomly if specificed, impute and scale the
    selected data, and return a pandas dataframe.

    Parameters
    ----------
    path: string
        path to 'Aspuru-Guzik_NSC_latent_representation_292D.csv'
    dtype: numpy type
        precision (data type) for reading float values
    ncols : int or None
        number of columns (drug latent representations) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply

    """
    df = pd.read_csv(path, engine='c', converters={'NSC': str}, dtype=dtype)
    df1 = pd.DataFrame(df.loc[:, 'NSC'])
    df2 = df.drop('NSC', axis=1)
    total = df2.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:, usecols]
    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(dtype)
    df = pd.concat([df1, df2], axis=1)
    return df
