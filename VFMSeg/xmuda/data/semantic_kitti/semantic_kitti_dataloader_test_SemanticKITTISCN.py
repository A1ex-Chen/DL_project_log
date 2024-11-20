def test_SemanticKITTISCN():
    from xmuda.data.utils.visualize import draw_points_image_labels, draw_bird_eye_view
    preprocess_dir = (
        '/datasets_local/datasets_mjaritz/semantic_kitti_preprocess/preprocess'
        )
    semantic_kitti_dir = (
        '/datasets_local/datasets_mjaritz/semantic_kitti_preprocess')
    pselab_paths = (
        '/home/docker_user/workspace/outputs/xmuda_journal/a2d2_semantic_kitti/fusion/fusion_xmuda_kl0.1_0.01/pselab_data/val.npy'
        ,)
    split = 'val',
    dataset = SemanticKITTISCN(split=split, preprocess_dir=preprocess_dir,
        semantic_kitti_dir=semantic_kitti_dir, pselab_paths=pselab_paths,
        merge_classes_style='A2D2', noisy_rot=0.1, flip_y=0.5, rot_z=2 * np
        .pi, transl=True, crop_size=(480, 302), bottom_crop=True, fliplr=
        0.5, color_jitter=(0.4, 0.4, 0.4))
    for i in (5 * [0]):
        data = dataset[i]
        coords = data['coords']
        seg_label = data['seg_label']
        img = np.moveaxis(data['img'], 0, 2)
        img_indices = data['img_indices']
        pseudo_label_2d = data['pseudo_label_2d']
        draw_points_image_labels(img, img_indices, seg_label,
            color_palette_type='SemanticKITTI', point_size=1)
        draw_points_image_labels(img, img_indices, pseudo_label_2d,
            color_palette_type='SemanticKITTI', point_size=1)
        assert len(pseudo_label_2d) == len(seg_label)
        draw_bird_eye_view(coords)
