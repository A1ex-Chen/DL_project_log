def append_nbrs(x, nbrs, num_nbrs):
    """
    Appends the neighbors to each molecule in the frame

    Args:
    x: data of shape (num_molecules, num_beads*features)
    nbrs: neighbor index of molecules of shape (num_molecules, 100)
    num_nbrs: int, number of neighbors to append

    Returns:
    x_wNbrs: concatenated features of all neighbors of shape (num_molecules, (num_nbrs+1)*num_beads*num_feature)
    """
    new_x_shape = np.array(x.shape)
    new_x_shape[1] *= num_nbrs + 1
    x_wNbrs = np.zeros(new_x_shape)
    for i in range(len(x)):
        nb_indices = nbrs[i, :num_nbrs + 1].astype(int)
        if not i:
            print('nbrs indices: ', nb_indices)
        nb_indices = nb_indices[nb_indices != -1]
        temp_mols = x[nb_indices]
        newshape = 1, np.prod(temp_mols.shape)
        temp_mols = np.reshape(temp_mols, newshape)
        x_wNbrs[i, :temp_mols.shape[1]] = temp_mols
    return x_wNbrs
