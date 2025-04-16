def eval_correspondences_mesh(self, mesh_files, pcl_tgt,
    project_to_final_mesh=False):
    """ Calculates correspondence score for meshes.

        Args:
            mesh_files (list): list of mesh files
            pcl_tgt (list): list of target point clouds
            project_to_final_mesh (bool): whether to project predictions to
                GT mesh by finding its NN in the target point cloud
        """
    mesh_t0 = load_mesh(mesh_files[0])
    mesh_pts_t0 = mesh_t0.vertices
    mesh_pts_t0 = np.expand_dims(mesh_pts_t0.astype(np.float32), axis=0)
    ind, _ = get_nearest_neighbors_indices_batch(mesh_pts_t0, np.
        expand_dims(pcl_tgt[0], axis=0))
    ind = ind[0].astype(int)
    eval_dict = {}
    for i in range(len(pcl_tgt)):
        v_t = load_mesh(mesh_files[i]).vertices
        pc_nn_t = pcl_tgt[i][ind]
        if project_to_final_mesh and i == len(pcl_tgt) - 1:
            ind2, _ = get_nearest_neighbors_indices_batch(np.expand_dims(
                v_t, axis=0).astype(np.float32), np.expand_dims(pcl_tgt[i],
                axis=0))
            v_t = pcl_tgt[i][ind2[0]]
        l2_loss = np.mean(np.linalg.norm(v_t - pc_nn_t, axis=-1)).item()
        eval_dict['l2 %d (mesh)' % i] = l2_loss
    return eval_dict
