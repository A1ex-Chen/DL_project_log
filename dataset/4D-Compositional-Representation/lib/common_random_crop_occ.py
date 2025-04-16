def random_crop_occ(points, pcl, n_kernel=3, radius=0.2):
    """ Randomly crop point cloud.
        Args:
            points (numpy array): Query points in the occupancy volume, (n, 3)
            pcl (numpy array): Point cloud corresponding to the occupancy volume, (n, 3)
            n_kernel (int): number of the removing centers
            radius (float): radius of the sphere mask
    """
    bbox = [pcl.min(0), pcl.max(0)]
    x = np.random.rand(n_kernel) * bbox[1][0] * 2 - bbox[1][0]
    y = np.random.rand(n_kernel) * bbox[1][1] * 2 - bbox[1][1]
    z = np.random.rand(n_kernel) * bbox[1][2] * 2 - bbox[1][2]
    centers = np.vstack([x, y, z]).transpose(1, 0)
    n_pts = points.shape[0]
    pcl_k = np.tile(points[:, None, :], [1, n_kernel, 1])
    center = np.tile(centers[None, ...], [n_pts, 1, 1])
    dist = np.linalg.norm(pcl_k - center, 2, axis=-1)
    mask = (dist > radius).sum(-1) >= n_kernel
    return mask, centers, radius
