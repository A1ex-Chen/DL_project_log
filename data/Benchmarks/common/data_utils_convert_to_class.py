def convert_to_class(y_one_hot, dtype=int):
    """Converts a one-hot class encoding (array with as many positions as total
    classes, with 1 in the corresponding class position, 0 in the other positions),
    or soft-max class encoding (array with as many positions as total
    classes, whose largest valued position is used as class membership)
    to an integer class encoding.

    Parameters
    ----------
    y_one_hot : numpy array
        Input array with one-hot or soft-max class encoding.
    dtype : data type
        Data type to use for the output numpy array.
        (Default: int, integer data is used to represent the
        class membership).

    Return
    ----------
    Returns a numpy array with an integer class encoding.
    """

    def maxi(a):
        return a.argmax()

    def iter_to_na(i):
        return np.fromiter(i, dtype=dtype)
    return np.array([maxi(a) for a in y_one_hot])
