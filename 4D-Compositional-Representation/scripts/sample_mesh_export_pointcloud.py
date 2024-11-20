def export_pointcloud(identity, motion, model_files, smpl_faces, args):
    out_folder = os.path.join(args.out_folder, 'D-FAUST', identity, motion,
        args.pointcloud_folder)
    if os.path.exists(out_folder):
        if not args.overwrite:
            print('Pointcloud already exist: %s' % out_folder)
            return
        else:
            shutil.rmtree(out_folder)
    os.makedirs(out_folder)
    all_vers = np.load(model_files)
    mesh = trimesh.Trimesh(all_vers[0].squeeze(), smpl_faces.squeeze(),
        process=False)
    _, face_idx = mesh.sample(args.pointcloud_size, return_index=True)
    alpha = np.random.dirichlet((1,) * 3, args.pointcloud_size)
    for it, verts in enumerate(all_vers):
        out_file = os.path.join(out_folder, '%08d.npz' % it)
        mesh = trimesh.Trimesh(verts.squeeze(), smpl_faces.squeeze(),
            process=False)
        loc = np.zeros(3)
        scale = np.array([1.0])
        vertices = mesh.vertices
        faces = mesh.faces
        v = vertices[faces[face_idx]]
        points = (alpha[:, :, None] * v).sum(axis=1)
        print('Writing pointcloud: %s' % out_file)
        if args.float16:
            dtype = np.float16
        else:
            dtype = np.float32
        points = points.astype(dtype)
        loc = loc.astype(dtype)
        scale = scale.astype(dtype)
        np.savez(out_file, points=points, loc=loc, scale=scale)
