@numba.njit
def group_transform_(loc_noise, rot_noise, locs, rots, group_center, valid_mask
    ):
    num_try = loc_noise.shape[1]
    r = 0.0
    x = 0.0
    y = 0.0
    rot_center = 0.0
    for i in range(loc_noise.shape[0]):
        if valid_mask[i]:
            x = locs[i, 0] - group_center[i, 0]
            y = locs[i, 1] - group_center[i, 1]
            r = np.sqrt(x ** 2 + y ** 2)
            rot_center = np.arctan2(x, y)
            for j in range(num_try):
                loc_noise[i, j, 0] += r * (np.sin(rot_center + rot_noise[i,
                    j]) - np.sin(rot_center))
                loc_noise[i, j, 1] += r * (np.cos(rot_center + rot_noise[i,
                    j]) - np.cos(rot_center))
