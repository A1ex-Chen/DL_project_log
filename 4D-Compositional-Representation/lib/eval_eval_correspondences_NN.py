def eval_correspondences_NN(self, vertices, pcl_inp, n_pts=10000):
    """ Evaluates correspondences with nearest neighbor algorithm.
        The vertices of time time 0 are taken, and then the predictions for
        later time steps are the NNs for these vertices in the later point
        clouds.

        Args:
            vertices (numpy array): vertices of size L x N_v x 3
            pcl_inp (numpy array): random point clouds of size L x N_pcl x 3
            n_pts (int): how many points should be used from the point clouds
        """
    n_t, n_pt = pcl_inp.shape[:2]
    v = np.expand_dims(vertices[0], axis=0)
    eval_dict = {}
    for i in range(n_t):
        pcl = pcl_inp[i, np.random.randint(n_pt, size=n_pts), :]
        ind, _ = get_nearest_neighbors_indices_batch(v, np.expand_dims(pcl,
            axis=0))[0]
        pred_v = pcl[ind]
        l2_loss = np.mean(np.linalg.norm(pred_v - vertices[i], axis=-1)).item()
        eval_dict['l2 %d (mesh)' % i] = l2_loss
    return eval_dict
