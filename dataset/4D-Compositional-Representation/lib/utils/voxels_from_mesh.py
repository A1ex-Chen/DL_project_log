@classmethod
def from_mesh(cls, mesh, resolution, loc=None, scale=None, method='ray'):
    bounds = mesh.bounds
    if loc is None:
        loc = (bounds[0] + bounds[1]) / 2
    if scale is None:
        scale = (bounds[1] - bounds[0]).max() / 0.9
    loc = np.asarray(loc)
    scale = float(scale)
    mesh = mesh.copy()
    mesh.apply_translation(-loc)
    mesh.apply_scale(1 / scale)
    if method == 'ray':
        voxel_data = voxelize_ray(mesh, resolution)
    elif method == 'fill':
        voxel_data = voxelize_fill(mesh, resolution)
    voxels = cls(voxel_data, loc, scale)
    return voxels
