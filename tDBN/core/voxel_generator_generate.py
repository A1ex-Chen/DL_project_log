def generate(self, points, max_voxels):
    return points_to_voxel(points, self._voxel_size, self.
        _point_cloud_range, self._max_num_points, True, max_voxels)
