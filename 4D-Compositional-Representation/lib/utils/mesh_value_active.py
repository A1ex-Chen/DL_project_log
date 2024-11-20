@property
def value_active(self):
    value_active = np.full(self.values.shape, False)
    value_active[:-1, :-1, :-1] |= self.voxel_active
    value_active[:-1, :-1, 1:] |= self.voxel_active
    value_active[:-1, 1:, :-1] |= self.voxel_active
    value_active[:-1, 1:, 1:] |= self.voxel_active
    value_active[1:, :-1, :-1] |= self.voxel_active
    value_active[1:, :-1, 1:] |= self.voxel_active
    value_active[1:, 1:, :-1] |= self.voxel_active
    value_active[1:, 1:, 1:] |= self.voxel_active
    return value_active
