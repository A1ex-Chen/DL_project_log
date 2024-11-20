def get_mesh_loc_scale(mesh, apply_scale=False):
    """ Loads and scales a mesh.

    Args:
        mesh_path (trimesh): trimesh
        loc (tuple): location
        scale (float): scaling factor
    """
    bbox = mesh.bounding_box.bounds
    loc = (bbox[0] + bbox[1]) / 2
    scale = (bbox[1] - bbox[0]).max()
    if apply_scale:
        mesh.apply_translation(-loc)
        mesh.apply_scale(1 / scale)
    return mesh, loc, scale, bbox
