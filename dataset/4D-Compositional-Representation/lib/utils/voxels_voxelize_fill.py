def voxelize_fill(mesh, resolution):
    bounds = mesh.bounds
    if (np.abs(bounds) >= 0.5).any():
        raise ValueError(
            'voxelize fill is only supported if mesh is inside [-0.5, 0.5]^3/')
    occ = voxelize_surface(mesh, resolution)
    occ = ndimage.morphology.binary_fill_holes(occ)
    return occ
