def voxelize_interior(mesh, resolution):
    shape = (resolution,) * 3
    bb_min = (0.5,) * 3
    bb_max = (resolution - 0.5,) * 3
    points = make_3d_grid(bb_min, bb_max, shape=shape).numpy()
    points = points + 0.1 * (np.random.rand(*points.shape) - 0.5)
    points = points / resolution - 0.5
    occ = check_mesh_contains(mesh, points)
    occ = occ.reshape(shape)
    return occ
