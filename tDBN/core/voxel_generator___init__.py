def __init__(self, voxel_size, point_cloud_range, max_num_points,
    max_voxels=20000):
    point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
    voxel_size = np.array(voxel_size, dtype=np.float32)
    grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
    grid_size = np.round(grid_size).astype(np.int64)
    self._voxel_size = voxel_size
    self._point_cloud_range = point_cloud_range
    self._max_num_points = max_num_points
    self._max_voxels = max_voxels
    self._grid_size = grid_size
