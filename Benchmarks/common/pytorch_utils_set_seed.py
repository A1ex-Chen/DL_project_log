def set_seed(seed):
    """Set the random number seed to the desired value

    Parameters
    ----------
    seed : integer
        Random number seed.
    """
    set_seed_defaultUtils(seed)
    torch.manual_seed(seed)
