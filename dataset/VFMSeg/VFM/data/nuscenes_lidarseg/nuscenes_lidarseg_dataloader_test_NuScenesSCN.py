def test_NuScenesSCN():
    from xmuda.data.utils.visualize import draw_points_image_labels, draw_points_image_depth, draw_bird_eye_view
    preprocess_dir = (
        '/datasets_local/datasets_mjaritz/nuscenes_lidarseg_preprocess/preprocess'
        )
    nuscenes_dir = '/datasets_local/datasets_mjaritz/nuscenes_preprocess'
    split = 'test_singapore',
    dataset = NuScenesLidarSegSCN(split=split, preprocess_dir=
        preprocess_dir, nuscenes_dir=nuscenes_dir, merge_classes=True,
        noisy_rot=0.1, flip_x=0.5, rot_z=2 * np.pi, transl=True, fliplr=0.5,
        color_jitter=(0.4, 0.4, 0.4))
    for i in [10, 20, 30, 40, 50, 60]:
        data = dataset[i]
        coords = data['coords']
        seg_label = data['seg_label']
        img = np.moveaxis(data['img'], 0, 2)
        img_indices = data['img_indices']
        draw_points_image_labels(img, img_indices, seg_label,
            color_palette_type='NuScenesLidarSeg', point_size=3)
        print('Number of points:', len(coords))
