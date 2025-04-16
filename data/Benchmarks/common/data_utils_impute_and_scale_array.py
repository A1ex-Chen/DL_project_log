def impute_and_scale_array(mat, scaling=None):
    """ Impute missing values with mean and scale data included in numpy array.

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
        Returns the numpy array imputed with the mean value of the         column and scaled by the method specified. If no scaling method is specified,         it returns the imputed numpy array.
    """
    imputer = Imputer(strategy='mean', copy=False)
    imputer.fit_transform(mat)
    return scale_array(mat, scaling)
