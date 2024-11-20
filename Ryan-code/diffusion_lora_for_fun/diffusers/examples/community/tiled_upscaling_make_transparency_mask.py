def make_transparency_mask(size, overlap_pixels, remove_borders=[]):
    size_x = size[0] - overlap_pixels * 2
    size_y = size[1] - overlap_pixels * 2
    for letter in ['l', 'r']:
        if letter in remove_borders:
            size_x += overlap_pixels
    for letter in ['t', 'b']:
        if letter in remove_borders:
            size_y += overlap_pixels
    mask = np.ones((size_y, size_x), dtype=np.uint8) * 255
    mask = np.pad(mask, mode='linear_ramp', pad_width=overlap_pixels,
        end_values=0)
    if 'l' in remove_borders:
        mask = mask[:, overlap_pixels:mask.shape[1]]
    if 'r' in remove_borders:
        mask = mask[:, 0:mask.shape[1] - overlap_pixels]
    if 't' in remove_borders:
        mask = mask[overlap_pixels:mask.shape[0], :]
    if 'b' in remove_borders:
        mask = mask[0:mask.shape[0] - overlap_pixels, :]
    return mask
