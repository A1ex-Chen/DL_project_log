def test_A2D2SCN():
    from xmuda.data.utils.visualize import draw_points_image_labels, draw_bird_eye_view
    preprocess_dir = '/datasets_local/datasets_mjaritz/a2d2_preprocess'
    split = 'test',
    dataset = A2D2SCN(split=split, preprocess_dir=preprocess_dir,
        merge_classes=True, noisy_rot=0.1, flip_y=0.5, rot_z=2 * np.pi,
        transl=True, fliplr=0.5, color_jitter=(0.4, 0.4, 0.4))
    for i in (5 * [0]):
        data = dataset[i]
        coords = data['coords']
        seg_label = data['seg_label']
        img = np.moveaxis(data['img'], 0, 2)
        img_indices = data['img_indices']
        draw_points_image_labels(img, img_indices, seg_label,
            color_palette_type='SemanticKITTI', point_size=3)
        draw_bird_eye_view(coords)
