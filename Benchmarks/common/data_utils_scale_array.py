def scale_array(mat, scaling=None):
    """ Scale data included in numpy array.

        Parameters
        ----------
        mat : numpy array
            Array to scale
        scaling : string
            String describing type of scaling to apply.
            Options recognized: 'maxabs', 'minmax', 'std'.
            'maxabs' : scales data to range [-1 to 1].
            'minmax' : scales data to range [-1 to 1].
            'std'    : scales data to normal variable with mean 0 and standard deviation 1.
            (Default: None, no scaling).

        Return
        ----------
        Returns the numpy array scaled by the method specified.         If no scaling method is specified, it returns the numpy         array unmodified.
    """
    if scaling is None or scaling.lower() == 'none':
        return mat
    if scaling == 'maxabs':
        scaler = MaxAbsScaler(copy=False)
    elif scaling == 'minmax':
        scaler = MinMaxScaler(copy=False)
    else:
        scaler = StandardScaler(copy=False)
    return scaler.fit_transform(mat)
