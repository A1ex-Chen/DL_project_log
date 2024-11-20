def __getitem__(self, index):
    data_dict = self.data[index]
    points = data_dict['points'].copy()
    seg_label = data_dict['seg_labels'].astype(np.int64)
    if self.label_mapping is not None:
        seg_label = self.label_mapping[seg_label]
    vfm_dict = []
    with open(osp.join(self.vfm_data_paths, str(index) + '.pkl'), 'rb') as f:
        vfm_dict.extend(pickle.load(f))
    out_dict = {}
    points_img = data_dict['points_img'].copy()
    img_path = osp.join(self.preprocess_dir, data_dict['camera_path'])
    image = Image.open(img_path)
    masks_sam = vfm_dict[-1]['sam']
    masks_seem = vfm_dict[-1]['seem']
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
    out_dict['img'] = torch.from_numpy(out_dict['img'])
    out_dict['img_indices'] = img_indices
    coords = augment_and_scale_3d(points, self.scale, self.full_scale,
        noisy_rot=self.noisy_rot, flip_y=self.flip_y, rot_z=self.rot_z,
        transl=self.transl)
    coords = coords.astype(np.int64)
    idxs = (coords.min(1) >= 0) * (coords.max(1) < self.full_scale)
    out_dict['coords'] = torch.from_numpy(coords[idxs])
    out_dict['feats'] = torch.from_numpy(np.ones([len(idxs), 1], np.float32))
    out_dict['seg_label'] = torch.from_numpy(seg_label[idxs])
    out_dict['img_indices'] = out_dict['img_indices'][idxs]
    """
        Prepare Mixed Data
        """
    out_dict['masks_seem'] = torch.from_numpy(np.stack(masks_seem))
    sampled_sam_mask, sampled_sam_indices = sample_mix_masks(masks_sam,
        out_dict['img_indices'])
    cut_mix_mask, cut_mix_indices = get_cut_masks(out_dict['img'], out_dict
        ['img_indices'])
    out_dict['sampled_sam_mask'] = torch.from_numpy(sampled_sam_mask)
    out_dict['sampled_sam_sel_indices'] = sampled_sam_indices
    out_dict['sam_mix_image'] = out_dict['img'].clone().permute(1, 2, 0)
    out_dict['sam_mix_label_2d'] = out_dict['seg_label'].clone()
    out_dict['sam_mix_indices'] = out_dict['img_indices']
    out_dict['sam_label'] = out_dict['seg_label'].clone()[sampled_sam_indices]
    out_dict['sam_mix_coords'] = out_dict['coords'].clone()
    out_dict['sam_mix_feats'] = out_dict['feats'].clone()
    out_dict['sam_mix_label_3d'] = out_dict['seg_label'].clone()
    out_dict['cut_mask'] = torch.from_numpy(cut_mix_mask)
    out_dict['cut_sel_indices'] = cut_mix_indices
    out_dict['cut_mix_image'] = out_dict['img'].clone().permute(1, 2, 0)
    out_dict['cut_mix_label_2d'] = out_dict['seg_label'].clone()
    out_dict['cut_mix_indices'] = out_dict['img_indices']
    out_dict['cut_label'] = out_dict['seg_label'].clone()[cut_mix_indices]
    out_dict['cut_mix_coords'] = out_dict['coords'].clone()
    out_dict['cut_mix_feats'] = out_dict['feats'].clone()
    out_dict['cut_mix_label_3d'] = out_dict['seg_label'].clone()
    out_dict['img'] = out_dict['img'].permute(1, 2, 0)
    return out_dict
