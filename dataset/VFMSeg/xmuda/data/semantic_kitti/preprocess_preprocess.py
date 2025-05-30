def preprocess(split_name, root_dir, out_dir):
    pkl_data = []
    split = getattr(splits, split_name)
    dataloader = DataLoader(DummyDataset(root_dir, split), num_workers=8)
    num_skips = 0
    for i, data_dict in enumerate(dataloader):
        if not data_dict:
            print('empty dict, continue')
            num_skips += 1
            continue
        for k, v in data_dict.items():
            data_dict[k] = v[0]
        print('{}/{} {}'.format(i, len(dataloader), data_dict['lidar_path']))
        lidar_path = data_dict['lidar_path'].replace(root_dir + '/', '')
        cam_path = data_dict['camera_path'].replace(root_dir + '/', '')
        seg_labels = data_dict.get('seg_label', None)
        if seg_labels is not None:
            seg_labels = seg_labels.numpy()
        out_dict = {'points': data_dict['points'].numpy(), 'seg_labels':
            seg_labels, 'points_img': data_dict['points_img'].numpy(),
            'lidar_path': lidar_path, 'camera_path': cam_path, 'image_size':
            tuple(data_dict['image_size'].numpy())}
        pkl_data.append(out_dict)
    print('Skipped {} files'.format(num_skips))
    save_dir = osp.join(out_dir, 'preprocess')
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, '{}.pkl'.format(split_name))
    with open(save_path, 'wb') as f:
        pickle.dump(pkl_data, f)
        print('Wrote preprocessed data to ' + save_path)
