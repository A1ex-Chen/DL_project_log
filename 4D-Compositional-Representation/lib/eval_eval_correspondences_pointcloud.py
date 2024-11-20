def eval_correspondences_pointcloud(self, pcl_pred_files, pcl_tgt):
    """ Calculates correspondence score for point clouds.

        Args:
            pcl_pred_files (list): list of point cloud prediction files
            pcl_tgt (list): list of target point clouds
        """
    pc_t0 = np.expand_dims(load_pointcloud(pcl_pred_files[0]), axis=0)
    ind, _ = get_nearest_neighbors_indices_batch(pc_t0, np.expand_dims(
        pcl_tgt[0], axis=0))
    ind = ind[0]
    eval_dict = {}
    for i in range(len(pcl_tgt)):
        pc_t = load_pointcloud(pcl_pred_files[i])
        pc_nn_t = pcl_tgt[i][ind]
        l2_loss = np.mean(np.linalg.norm(pc_t - pc_nn_t, axis=-1)).item()
        eval_dict['l2 %d (pcl)' % i] = l2_loss
    return eval_dict
