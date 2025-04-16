def load_csv_data(train_path, test_path=None, sep=',', nrows=None, x_cols=
    None, y_cols=None, drop_cols=None, onehot_cols=None, n_cols=None,
    random_cols=False, shuffle=False, scaling=None, dtype=None,
    validation_split=None, return_dataframe=True, return_header=False, seed
    =DEFAULT_SEED):
    """ Load data from the files specified.
        Columns corresponding to data features and labels can be specified. A one-hot
        encoding can be used for either features or labels.
        If validation_split is specified, trainig data is further split into training
        and validation partitions.
        pandas DataFrames are used to load and pre-process the data. If specified,
        those DataFrames are returned. Otherwise just values are returned.
        Labels to output can be integer labels (for classification) or
        continuous labels (for regression).
        Columns to load can be specified, randomly selected or a subset can be dropped.
        Order of rows can be shuffled. Data can be rescaled.
        This function assumes that the files contain a header with column names.

        Parameters
        ----------
        train_path : filename
            Name of the file to load the training data.
        test_path : filename
            Name of the file to load the testing data. (Optional).
        sep : character
            Character used as column separator.
            (Default: ',', comma separated values).
        nrows : integer
            Number of rows to load from the files.
            (Default: None, all the rows are used).
        x_cols : list
            List of columns to use as features.
            (Default: None).
        y_cols : list
            List of columns to use as labels.
            (Default: None).
        drop_cols : list
            List of columns to drop from the files being loaded.
            (Default: None, all the columns are used).
        onehot_cols : list
            List of columns to one-hot encode.
            (Default: None).
        n_cols : integer
            Number of columns to load from the files.
            (Default: None).
        random_cols : boolean
            Boolean flag to indicate random selection of columns.
            If True a number of n_cols columns is randomly selected, if False
            the specified columns are used.
            (Default: False).
        shuffle : boolean
            Boolean flag to indicate row shuffling. If True the rows are
            re-ordered, if False the order in which rows are read is
            preserved.
            (Default: False, no permutation of the loading row order).
        scaling : string
            String describing type of scaling to apply.
            Options recognized: 'maxabs', 'minmax', 'std'.
            'maxabs' : scales data to range [-1 to 1].
            'minmax' : scales data to range [-1 to 1].
            'std'    : scales data to normal variable with mean 0 and standard deviation 1.
            (Default: None, no scaling).
        dtype : data type
            Data type to use for the output pandas DataFrames.
            (Default: None).
        validation_split : float
            Fraction of training data to set aside for validation.
            (Default: None, no validation partition is constructed).
        return_dataframe : boolean
            Boolean flag to indicate that the pandas DataFrames
            used for data pre-processing are to be returned.
            (Default: True, pandas DataFrames are returned).
        return_header : boolean
            Boolean flag to indicate if the column headers are
            to be returned.
            (Default: False, no column headers are separetely returned).
        seed : int
            Value to intialize or re-seed the generator.
            (Default: DEFAULT_SEED defined in default_utils).


        Return
        ----------
        Tuples of data features and labels are returned, for         train, validation and testing partitions, together with the column         names (headers). The specific objects to return depend         on the options selected.
    """
    if x_cols is None and drop_cols is None and n_cols is None:
        usecols = None
        y_names = None
    else:
        df_cols = pd.read_csv(train_path, engine='c', sep=sep, nrows=0)
        df_x_cols = df_cols.copy()
        if y_cols is not None:
            df_x_cols = df_x_cols.drop(df_cols[y_cols], axis=1)
        if drop_cols is not None:
            df_x_cols = df_x_cols.drop(df_cols[drop_cols], axis=1)
        reserved = []
        if onehot_cols is not None:
            reserved += onehot_cols
        if x_cols is not None:
            reserved += x_cols
        nx = df_x_cols.shape[1]
        if n_cols and n_cols < nx:
            if random_cols:
                indexes = sorted(np.random.choice(list(range(nx)), n_cols,
                    replace=False))
            else:
                indexes = list(range(n_cols))
            x_names = list(df_x_cols[indexes])
            unreserved = [x for x in x_names if x not in reserved]
            n_keep = np.maximum(n_cols - len(reserved), 0)
            combined = reserved + unreserved[:n_keep]
            x_names = [x for x in df_x_cols if x in combined]
        elif x_cols is not None:
            x_names = list(df_x_cols[x_cols])
        else:
            x_names = list(df_x_cols.columns)
        usecols = x_names
        if y_cols is not None:
            y_names = list(df_cols[y_cols])
            usecols = y_names + x_names
    df_train = pd.read_csv(train_path, engine='c', sep=sep, nrows=nrows,
        usecols=usecols)
    if test_path:
        df_test = pd.read_csv(test_path, engine='c', sep=sep, nrows=nrows,
            usecols=usecols)
    else:
        df_test = df_train[0:0].copy()
    if y_cols is None:
        y_names = []
    elif y_names is None:
        y_names = list(df_train[0:0][y_cols])
    if shuffle:
        df_train = df_train.sample(frac=1, random_state=seed)
        if test_path:
            df_test = df_test.sample(frac=1, random_state=seed)
    df_cat = pd.concat([df_train, df_test])
    df_y = df_cat[y_names]
    df_x = df_cat.drop(y_names, axis=1)
    if onehot_cols is not None:
        for col in onehot_cols:
            if col in y_names:
                df_dummy = pd.get_dummies(df_y[col], prefix=col, prefix_sep=':'
                    )
                df_y = pd.concat([df_dummy, df_y.drop(col, axis=1)], axis=1)
            else:
                df_dummy = pd.get_dummies(df_x[col], prefix=col, prefix_sep=':'
                    )
                df_x = pd.concat([df_dummy, df_x.drop(col, axis=1)], axis=1)
    if scaling is not None:
        mat = scale_array(df_x.values, scaling)
        df_x = pd.DataFrame(mat, index=df_x.index, columns=df_x.columns,
            dtype=dtype)
    n_train = df_train.shape[0]
    x_train = df_x[:n_train]
    y_train = df_y[:n_train]
    x_test = df_x[n_train:]
    y_test = df_y[n_train:]
    return_y = y_cols is not None
    return_val = (validation_split and validation_split > 0 and 
        validation_split < 1)
    return_test = test_path
    if return_val:
        n_val = int(n_train * validation_split)
        x_val = x_train[-n_val:]
        y_val = y_train[-n_val:]
        x_train = x_train[:-n_val]
        y_train = y_train[:-n_val]
    ret = [x_train]
    ret = ret + [y_train] if return_y else ret
    ret = ret + [x_val] if return_val else ret
    ret = ret + [y_val] if return_y and return_val else ret
    ret = ret + [x_test] if return_test else ret
    ret = ret + [y_test] if return_y and return_test else ret
    if not return_dataframe:
        ret = [x.values for x in ret]
    if return_header:
        ret = ret + [df_x.columns.tolist(), df_y.columns.tolist()]
    return tuple(ret) if len(ret) > 1 else ret
