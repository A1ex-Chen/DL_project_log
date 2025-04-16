def get_loc_scale(mesh, args):
    if not args.resize:
        loc = np.zeros(3)
        scale = 1.0
    else:
        bbox = mesh.bounding_box.bounds
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max() / (1 - args.bbox_padding)
    return loc, scale
