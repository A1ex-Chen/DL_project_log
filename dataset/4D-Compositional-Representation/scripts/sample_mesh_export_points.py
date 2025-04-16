def export_points(identity, motion, model_files, smpl_faces, args):
    out_folder = os.path.join(args.out_folder, 'D-FAUST', identity, motion,
        args.points_folder)
    if os.path.exists(out_folder):
        if not args.overwrite:
            print('Points already exist: %s' % out_folder)
            return
        else:
            shutil.rmtree(out_folder)
    os.makedirs(out_folder)
    all_vers = np.load(model_files)
    n_points_uniform = int(args.points_size * args.points_uniform_ratio)
    n_points_surface = args.points_size - n_points_uniform
    for it, verts in enumerate(all_vers):
        out_file = os.path.join(out_folder, '%08d.npz' % it)
        mesh = trimesh.Trimesh(verts.squeeze(), smpl_faces.squeeze(),
            process=False)
        if not mesh.is_watertight:
            print('Warning: mesh %s is not watertight!')
        loc_self, scale_self = get_loc_scale(mesh, args)
        loc_global = np.array([-0.005493, -0.1888, 0.07587])
        scale_global = np.array([2.338])
        mesh.apply_translation(-loc_global)
        mesh.apply_scale(1 / scale_global)
        boxsize = 1 + args.points_padding
        points_uniform = np.random.rand(n_points_uniform, 3)
        points_uniform = boxsize * (points_uniform - 0.5)
        points_uniform = (loc_self + scale_self * points_uniform - loc_global
            ) / scale_global
        points_surface = mesh.sample(n_points_surface)
        points_surface += args.points_sigma * np.random.randn(n_points_surface,
            3)
        points = np.concatenate([points_uniform, points_surface], axis=0)
        occupancies = check_mesh_contains(mesh, points)
        print('Writing points: %s' % out_file)
        if args.float16:
            dtype = np.float16
        else:
            dtype = np.float32
        points = points.astype(dtype)
        loc = loc_global.astype(dtype)
        scale = scale_global.astype(dtype)
        if args.packbits:
            occupancies = np.packbits(occupancies)
        np.savez(out_file, points=points, occupancies=occupancies, loc=loc,
            scale=scale)
