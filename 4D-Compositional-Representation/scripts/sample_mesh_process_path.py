def process_path(in_path, args):
    smpl_model = SMPLModel(model_path=
        'data/human_dataset/smpl_models/model_300_m.pkl')
    smpl_faces = smpl_model.faces
    identity, motion = in_path.split('/')[-2:]
    model_file = os.path.join(in_path, 'smpl_vers.npy')
    if args.pointcloud_folder is not None:
        export_pointcloud(identity, motion, model_file, smpl_faces, args)
    if args.points_folder is not None:
        export_points(identity, motion, model_file, smpl_faces, args)
    print(in_path)
