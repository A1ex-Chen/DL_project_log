def increase_resolution(self):
    self.resolution = 2 * self.resolution
    shape_values = (self.resolution + 1,) * 3
    value_known = np.full(shape_values, False)
    value_known[::2, ::2, ::2] = self.value_known
    values = upsample3d_nn(self.values)
    values = values[:-1, :-1, :-1]
    self.values = values
    self.value_known = value_known
    self.voxel_active = upsample3d_nn(self.voxel_active)
