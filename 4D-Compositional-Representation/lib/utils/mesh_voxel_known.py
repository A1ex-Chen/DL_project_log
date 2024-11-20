@property
def voxel_known(self):
    value_known = self.value_known
    voxel_known = voxels.check_voxel_occupied(value_known)
    return voxel_known
