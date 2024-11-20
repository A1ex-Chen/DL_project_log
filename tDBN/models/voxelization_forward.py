def forward(self, features, num_voxels):
    voxel_wise = num_voxels.type_as(features).view(-1, 1)
    return voxel_wise
