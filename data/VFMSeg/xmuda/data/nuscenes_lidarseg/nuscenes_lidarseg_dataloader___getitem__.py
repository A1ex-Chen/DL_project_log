def __getitem__(self, index):
    data_dict = self.data[index]
    points = data_dict['points'].copy()
    seg_label = data_dict['seg_labels'].astype(np.int64)
    if self.label_mapping is not None:
        seg_label = self.label_mapping[seg_label]
    out_dict = {}
    points_img = data_dict['points_img'].copy()
    img_path = osp.join(self.nuscenes_dir, data_dict['camera_path'])
    image = Image.open(img_path)
    if self.with_vfm:
        points_img_orig = data_dict['points_img'].copy()
        points_img_orig[:, 0] = np.floor(points_img_orig[:, 0])
        points_img_orig[:, 1] = np.floor(points_img_orig[:, 1])
        img_indices_org = points_img_orig.astype(np.int64)
        assert np.all(img_indices_org[:, 0] >= 0)
        assert np.all(img_indices_org[:, 1] >= 0)
        assert np.all(img_indices_org[:, 0] < image.size[1])
        assert np.all(img_indices_org[:, 1] < image.size[0])
        image_orig = image.copy()
    if self.resize:
        if not image.size == self.resize:
            assert image.size[0] > self.resize[0]
            points_img[:, 0] = float(self.resize[1]) / image.size[1
                ] * np.floor(points_img[:, 0])
            points_img[:, 1] = float(self.resize[0]) / image.size[0
                ] * np.floor(points_img[:, 1])
            image = image.resize(self.resize, Image.BILINEAR)
    img_indices = points_img.astype(np.int64)
    assert np.all(img_indices[:, 0] >= 0)
    assert np.all(img_indices[:, 1] >= 0)
    assert np.all(img_indices[:, 0] < image.size[1])
    assert np.all(img_indices[:, 1] < image.size[0])
    if self.color_jitter is not None:
        image = self.color_jitter(image)
    image = np.array(image, dtype=np.float32, copy=False) / 255.0
    if np.random.rand() < self.fliplr:
        image = np.ascontiguousarray(np.fliplr(image))
        img_indices[:, 1] = image.shape[1] - 1 - img_indices[:, 1]
    if self.image_normalizer:
        mean, std = self.image_normalizer
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        image = (image - mean) / std
    out_dict['img'] = np.moveaxis(image, -1, 0)
    out_dict['img_indices'] = img_indices
    if self.with_vfm:
        out_dict['img_paths'] = img_path
        out_dict['img_indices_orig'] = img_indices_org
        image_orig = np.array(image_orig, dtype=np.float32, copy=False) / 255.0
        out_dict['img_instances_orig'] = np.moveaxis(image_orig, -1, 0)
    coords = augment_and_scale_3d(points, self.scale, self.full_scale,
        noisy_rot=self.noisy_rot, flip_x=self.flip_x, rot_z=self.rot_z,
        transl=self.transl)
    coords = coords.astype(np.int64)
    idxs = (coords.min(1) >= 0) * (coords.max(1) < self.full_scale)
    out_dict['coords'] = coords[idxs]
    out_dict['feats'] = np.ones([len(idxs), 1], np.float32)
    out_dict['seg_label'] = seg_label[idxs]
    out_dict['img_indices'] = out_dict['img_indices'][idxs]
    if self.with_vfm:
        out_dict['img_indices_orig'] = out_dict['img_indices_orig'][idxs]
    if self.pselab_data is not None:
        out_dict['pseudo_label_2d'] = self.pselab_data[index]['pseudo_label_2d'
            ][idxs]
        if self.pselab_data[index]['pseudo_label_3d'] is None:
            out_dict['pseudo_label_3d'] = None
        else:
            out_dict['pseudo_label_3d'] = self.pselab_data[index][
                'pseudo_label_3d'][idxs]
    if self.output_orig:
        out_dict.update({'orig_seg_label': seg_label, 'orig_points_idx': idxs})
    return out_dict
