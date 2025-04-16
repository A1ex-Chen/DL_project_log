def set_seed(seed):
    """Set the seed of the pseudo-random generator to the specified value.

    Parameters
    ----------
    seed : int
        Value to intialize or re-seed the generator.
    """
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    random.seed(seed)
