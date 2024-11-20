def glob_frames(self, scenes):
    for scene in scenes:
        glob_path = osp.join(self.root_dir, 'dataset', 'sequences', scene,
            'image_2', '*.png')
        cam_paths = sorted(glob.glob(glob_path))
        calib = self.read_calib(osp.join(self.root_dir, 'dataset',
            'sequences', scene, 'calib.txt'))
        proj_matrix = calib['P2'] @ calib['Tr']
        proj_matrix = proj_matrix.astype(np.float32)
        for cam_path in cam_paths:
            basename = osp.basename(cam_path)
            frame_id = osp.splitext(basename)[0]
            assert frame_id.isdigit()
            data = {'camera_path': cam_path, 'lidar_path': osp.join(self.
                root_dir, 'dataset', 'sequences', scene, 'velodyne', 
                frame_id + '.bin'), 'label_path': osp.join(self.root_dir,
                'dataset', 'sequences', scene, 'labels', frame_id +
                '.label'), 'proj_matrix': proj_matrix}
            for k in ['camera_path', 'lidar_path']:
                if not osp.exists(data[k]):
                    raise IOError('File not found {}'.format(data[k]))
            self.data.append(data)
