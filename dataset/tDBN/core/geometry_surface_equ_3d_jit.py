@numba.jit(nopython=False)
def surface_equ_3d_jit(polygon_surfaces):
    surface_vec = polygon_surfaces[:, :, :2, :] - polygon_surfaces[:, :, 1:3, :
        ]
    normal_vec = np.cross(surface_vec[:, :, 0, :], surface_vec[:, :, 1, :])
    d = np.einsum('aij, aij->ai', normal_vec, polygon_surfaces[:, :, 0, :])
    return normal_vec, -d
