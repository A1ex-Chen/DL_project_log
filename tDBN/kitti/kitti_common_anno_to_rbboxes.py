def anno_to_rbboxes(anno):
    loc = anno['location']
    dims = anno['dimensions']
    rots = anno['rotation_y']
    rbboxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
    return rbboxes
