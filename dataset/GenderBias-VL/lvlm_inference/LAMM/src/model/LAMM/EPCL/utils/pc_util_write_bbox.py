def write_bbox(scene_bbox, out_filename):
    """Export scene bbox to meshes
    Args:
        scene_bbox: (N x 6 numpy array): xyz pos of center and 3 lengths
        out_filename: (string) filename

    Note:
        To visualize the boxes in MeshLab.
        1. Select the objects (the boxes)
        2. Filters -> Polygon and Quad Mesh -> Turn into Quad-Dominant Mesh
        3. Select Wireframe view.
    """

    def convert_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt
    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_box_to_trimesh_fmt(box))
    mesh_list = trimesh.util.concatenate(scene.dump())
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='ply')
    return
