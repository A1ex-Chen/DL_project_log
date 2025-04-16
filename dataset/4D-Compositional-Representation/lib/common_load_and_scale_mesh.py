def load_and_scale_mesh(mesh_path, loc=None, scale=None):
    """ Loads and scales a mesh.

    Args:
        mesh_path (str): mesh path
        loc (tuple): location
        scale (float): scaling factor
    """
    mesh = trimesh.load(mesh_path, process=False)
    if loc is None or scale is None:
        bbox = mesh.bounding_box.bounds
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max()
    mesh.apply_translation(-loc)
    mesh.apply_scale(1 / scale)
    return loc, scale, mesh
