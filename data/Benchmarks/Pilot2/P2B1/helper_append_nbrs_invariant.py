def append_nbrs_invariant(x, nbrs, num_nbrs):
    """
    Create a neighborhood feature vetor for each molecule in the frame
    Drops x, y, z values
    Stores distance relative to the center of mass of the head beads of molecule as the first (0) feature.
    Stores angles relative to the orientation of molecule as the second (1) feature.

    Args:
    x: data of shape (num_molecules, num_beads, features)
    nbrs: neighbor index of molecules of shape (num_molecules, 100)
    num_nbrs: int, number of neighbors to append

    Returns:
    x_wNbrs: concatenated features of all neighbors of shape (num_molecules, (num_nbrs+1)*num_beads*num_feature)
             The features are in the order:
             [relative_distance, relative_angle, 'CHOL', 'DPPC', 'DIPC', 'Head', 'Tail', 'BL1', 'BL2', 'BL3', 'BL4', 'BL5', 'BL6',
             'BL7', 'BL8', 'BL9', 'BL10', 'BL11', 'BL12']
    """
    new_x_shape = np.array((x.shape[0], x.shape[1] * (x.shape[2] - 1)))
    new_x_shape[1] *= num_nbrs + 1
    x_wNbrs = np.zeros(new_x_shape)
    for i in range(len(x)):
        nb_indices = nbrs[i, :num_nbrs + 1].astype(int)
        nb_indices = nb_indices[nb_indices != -1]
        temp_mols = x[nb_indices]
        xy_feats = np.copy(temp_mols[:, :, :2])
        temp_mols = temp_mols[:, :, 1:]
        com = get_com_head(x[i])
        orientation = np.squeeze(orientationVector(x[i, 3, :2].reshape(1, 1,
            -1), x[i, 2, :2], [1.0, 1.0]))
        temp_mols[:, :, 0] = periodicDistance(xy_feats, com[0, :2], [1.0, 1.0])
        temp_mols[:, :, 1] = get_angles(xy_feats, com[0, :2], orientation,
            [1.0, 1.0])
        ind = np.argwhere(temp_mols[:, 0, 2] == 1)
        temp_mols[ind, 8:, :] = 0
        sorted_arg = np.argsort(temp_mols[1:, 0, 1]) + 1
        temp_mols[1:, :, :] = temp_mols[sorted_arg, :, :]
        newshape = 1, np.prod(temp_mols.shape)
        temp_mols = np.reshape(temp_mols, newshape)
        x_wNbrs[i, :temp_mols.shape[1]] = temp_mols
    return x_wNbrs
