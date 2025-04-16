def voxelize_ray(mesh, resolution):
    occ_surface = voxelize_surface(mesh, resolution)
    occ_interior = voxelize_interior(mesh, resolution)
    occ = occ_interior | occ_surface
    return occ
