@property
def voxel_empty(self):
    occ = self.occupancies
    return ~voxels.check_voxel_boundary(occ)
