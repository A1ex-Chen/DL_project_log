def get_com(x):
    """
    Calculating the Center of Mass of the molecule.
    Using only the first 8 beads if the molecule is CHOL
    """
    if x[0, 3] == 1:
        return np.mean(x[:8, :3], axis=0)
    else:
        return np.mean(x[:, :3], axis=0)
