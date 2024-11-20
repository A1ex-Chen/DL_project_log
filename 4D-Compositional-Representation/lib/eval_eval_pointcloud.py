def eval_pointcloud(self, pointcloud, pointcloud_tgt, normals=None,
    normals_tgt=None):
    """ Evaluates a point cloud.

        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
        """
    if pointcloud.shape[0] == 0:
        logger.warn('Empty pointcloud / mesh detected!')
        out_dict = EMPTY_PCL_DICT.copy()
        if normals is not None and normals_tgt is not None:
            out_dict.update(EMPTY_PCL_DICT_NORMALS)
        return out_dict
    pointcloud = np.asarray(pointcloud)
    pointcloud_tgt = np.asarray(pointcloud_tgt)
    completeness, completeness_normals = distance_p2p(pointcloud_tgt,
        normals_tgt, pointcloud, normals)
    completeness2 = completeness ** 2
    completeness = completeness.mean()
    completeness2 = completeness2.mean()
    completeness_normals = completeness_normals.mean()
    accuracy, accuracy_normals = distance_p2p(pointcloud, normals,
        pointcloud_tgt, normals_tgt)
    accuracy2 = accuracy ** 2
    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()
    accuracy_normals = accuracy_normals.mean()
    chamfer = completeness2 + accuracy2
    normals_correctness = 0.5 * completeness_normals + 0.5 * accuracy_normals
    out_dict = {'completeness': completeness, 'accuracy': accuracy,
        'normals completeness': completeness_normals, 'normals accuracy':
        accuracy_normals, 'normals': normals_correctness, 'completeness2':
        completeness2, 'accuracy2': accuracy2, 'chamfer': chamfer}
    return out_dict
