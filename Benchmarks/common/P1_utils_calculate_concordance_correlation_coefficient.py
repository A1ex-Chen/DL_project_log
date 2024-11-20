def calculate_concordance_correlation_coefficient(u, v):
    """
    This function calculates the concordance correlation coefficient between two input 1-D numpy arrays.

    Parameters:
    -----------
    u: 1-D numpy array of a variable
    v: 1-D numpy array of a variable

    Returns:
    --------
    ccc: a numeric value of concordance correlation coefficient between the two input variables.
    """
    a = 2 * np.mean((u - np.mean(u)) * (v - np.mean(v)))
    b = np.mean(np.square(u - np.mean(u))) + np.mean(np.square(v - np.mean(v))
        ) + np.square(np.mean(u) - np.mean(v))
    ccc = a / b
    return ccc
