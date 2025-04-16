def glob_frames(self, scenes):
    for scene in scenes:
        cam_paths = sorted(glob.glob(osp.join(self.root_dir, scene,
            'camera', 'cam_front_center', '*.png')))
        for cam_path in cam_paths:
            basename = osp.basename(cam_path)
            datetime = basename[:14]
            assert datetime.isdigit()
            frame_id = basename[-13:-4]
            assert frame_id.isdigit()
            data = {'camera_path': cam_path, 'lidar_path': osp.join(self.
                root_dir, scene, 'lidar', 'cam_front_center', datetime +
                '_lidar_frontcenter_' + frame_id + '.npz'), 'label_path':
                osp.join(self.root_dir, scene, 'label', 'cam_front_center',
                datetime + '_label_frontcenter_' + frame_id + '.png')}
            for k, v in data.items():
                if not osp.exists(v):
                    raise IOError('File not found {}'.format(v))
            self.data.append(data)
