def get_cut_masks(img, indices, mask_props=0.32):
    _, h, w = img.shape
    y_props = np.exp(np.random.uniform(low=0.0, high=1, size=(1, 1)) * np.
        log(mask_props))
    x_props = mask_props / y_props
    sizes = np.round(np.stack([y_props, x_props], axis=2) * np.array((h, w)
        )[None, None, :])
    positions = np.round((np.array((h, w)) - sizes) * np.random.uniform(low
        =0.0, high=1.0, size=sizes.shape))
    rectangles = np.append(positions, positions + sizes, axis=2)
    mask = np.zeros((h, w), dtype=np.bool8)
    y0, x0, y1, x1 = rectangles[0][0]
    mask[int(y0):int(y1), int(x0):int(x1)] = True
    cut_2d_to_3d_indices = mask[indices[:, 0], indices[:, 1]] == True
    return mask, cut_2d_to_3d_indices
