def check_aspect(crop_range, aspect_min):
    xy_aspect = np.min(crop_range[:2]) / np.max(crop_range[:2])
    xz_aspect = np.min(crop_range[[0, 2]]) / np.max(crop_range[[0, 2]])
    yz_aspect = np.min(crop_range[1:]) / np.max(crop_range[1:])
    return (xy_aspect >= aspect_min or xz_aspect >= aspect_min or yz_aspect >=
        aspect_min)
