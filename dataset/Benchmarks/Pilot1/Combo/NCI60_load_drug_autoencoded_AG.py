def load_drug_autoencoded_AG(ncols=None, scaling='std', add_prefix=True):
    """Load drug latent representation from Aspuru-Guzik's variational
    autoencoder, sub-select columns of drugs randomly if specificed,
    impute and scale the selected data, and return a pandas dataframe

    Parameters
    ----------
    ncols : int or None
        number of columns (drug latent representations) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    add_prefix: True or False
        add feature namespace prefix
    """
    path = get_file(P1B3_URL +
        'Aspuru-Guzik_NSC_latent_representation_292D.csv')
    df = global_cache.get(path)
    if df is None:
        df = pd.read_csv(path, engine='c', dtype=np.float32)
        global_cache[path] = df
    df1 = pd.DataFrame(df.loc[:, 'NSC'].astype(int).astype(str))
    df2 = df.drop('NSC', axis=1)
    if add_prefix:
        df2 = df2.add_prefix('smiles_latent_AG.')
    total = df2.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:, usecols]
    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(np.float32)
    df = pd.concat([df1, df2], axis=1)
    return df
