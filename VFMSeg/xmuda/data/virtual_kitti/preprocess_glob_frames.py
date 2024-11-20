def glob_frames(self, scenes):
    for scene in scenes:
        glob_path = osp.join(self.root_dir, 'vkitti3d_npy', scene, '*.npy')
        lidar_paths = sorted(glob.glob(glob_path))
        for lidar_path in lidar_paths:
            if not osp.exists(lidar_path):
                raise IOError('File not found {}'.format(lidar_path))
            basename = osp.basename(lidar_path)
            frame_id = osp.splitext(basename)[0]
            assert frame_id.isdigit()
            data = {'lidar_path': lidar_path, 'scene_id': scene, 'frame_id':
                frame_id}
            self.data.append(data)
