def load_cellline_expressions(path, dtype, ncols=None, scaling='std'):
    """Load cell line expression data, sub-select columns of gene expression
        randomly if specificed, scale the selected data and return a
        pandas dataframe.

    Parameters
    ----------
    path: string
        path to 'RNA_5_Platform_Gene_Transcript_Averaged_intensities.transposed.txt'
    dtype: numpy type
        precision (data type) for reading float values
    ncols : int or None
        number of columns (gene expression) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    """
    df = pd.read_csv(path, sep='\t', engine='c', na_values=['na', '-', ''])
    df1 = df['CellLine']
    df1 = df1.map(lambda x: x.replace('.', ':'))
    df1.name = 'CELLNAME'
    df2 = df.drop('CellLine', axis=1)
    total = df2.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:, usecols]
    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(dtype)
    df = pd.concat([df1, df2], axis=1)
    return df
