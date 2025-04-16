def get_com_head(x):
    """
    Calculating the Center of Mass from head beads of the molecule.
    Using only the first 8 beads if the molecule is CHOL
    """
    head_inds = np.argwhere(x[:, 6] == 1)
    return np.mean(x[head_inds, :3], axis=0)
