def write_lines_as_cylinders(pcl, filename, rad=0.005, res=64):
    """Create lines represented as cylinders connecting pairs of 3D points
    Args:
        pcl: (N x 2 x 3 numpy array): N pairs of xyz pos
        filename: (string) filename for the output mesh (ply) file
        rad: radius for the cylinder
        res: number of sections used to create the cylinder
    """
    scene = trimesh.scene.Scene()
    for src, tgt in pcl:
        vec = tgt - src
        M = trimesh.geometry.align_vectors([0, 0, 1], vec, False)
        vec = tgt - src
        M[:3, 3] = 0.5 * src + 0.5 * tgt
        height = np.sqrt(np.dot(vec, vec))
        scene.add_geometry(trimesh.creation.cylinder(radius=rad, height=
            height, sections=res, transform=M))
    mesh_list = trimesh.util.concatenate(scene.dump())
    trimesh.io.export.export_mesh(mesh_list, '%s.ply' % filename, file_type
        ='ply')
