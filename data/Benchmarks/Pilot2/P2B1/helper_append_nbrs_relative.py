def append_nbrs_relative(x, nbrs, num_nbrs):
    """
    Appends the neighbors to each molecule in the frame
    Also, uses x, y, z positions relative to the center of mass of the molecule.

    Args:
    x: data of shape (num_molecules, num_beads, features)
    nbrs: neighbor index of molecules of shape (num_molecules, 100)
    num_nbrs: int, number of neighbors to append

    Returns:
    x_wNbrs: concatenated features of all neighbors of shape (num_molecules, (num_nbrs+1)*num_beads*num_feature)

    """
    new_x_shape = np.array((x.shape[0], np.prod(x.shape[1:])))
    new_x_shape[1] *= num_nbrs + 1
    x_wNbrs = np.zeros(new_x_shape)
    for i in range(len(x)):
        nb_indices = nbrs[i, :num_nbrs + 1].astype(int)
        nb_indices = nb_indices[nb_indices != -1]
        temp_mols = x[nb_indices]
        com = get_com(x[i])
        temp_mols = periodicVector(temp_mols, com, [1.0, 1.0, 0.3])
        ind = np.argwhere(temp_mols[:, 1, 3] == 1)
        temp_mols[ind, 8:, :] = 0
        newshape = 1, np.prod(temp_mols.shape)
        temp_mols = np.reshape(temp_mols, newshape)
        x_wNbrs[i, :temp_mols.shape[1]] = temp_mols
    return x_wNbrs
