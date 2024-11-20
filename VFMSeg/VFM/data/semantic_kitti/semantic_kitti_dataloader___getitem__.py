def __getitem__(self, index):
    data_dict = self.data[index]
    points = data_dict['points'].copy()
    seg_label = data_dict['seg_labels']
    if seg_label is not None:
        seg_label = seg_label.astype(np.int64)
    if self.label_mapping is not None and seg_label is not None:
        seg_label = self.label_mapping[seg_label]
    vfm_dict = []
    with open(osp.join(self.vfm_data_paths, str(index) + '.pkl'), 'rb') as f:
        vfm_dict.extend(pickle.load(f))
    masks_sam = vfm_dict[-1]['sam']
    masks_seem = vfm_dict[-1]['seem']
    out_dict = {}
    keep_idx = np.ones(len(points), dtype=np.bool)
    points_img = data_dict['points_img'].copy()
    img_path = osp.join(self.semantic_kitti_dir, data_dict['camera_path'])
    image = Image.open(img_path)
    if self.crop_size:
        valid_crop = False
        for _ in range(10):
            if self.bottom_crop:
                left = int(np.random.rand() * (image.size[0] + 1 - self.
                    crop_size[0]))
                right = left + self.crop_size[0]
                top = image.size[1] - self.crop_size[1]
                bottom = image.size[1]
            elif len(self.rand_crop) > 0:
                crop_height, crop_width = self.rand_crop[0::2
                    ] + np.random.rand(2) * (self.rand_crop[1::2] - self.
                    rand_crop[0::2])
                top = np.random.rand() * (1 - crop_height) * image.size[1]
                left = np.random.rand() * (1 - crop_width) * image.size[0]
                bottom = top + crop_height * image.size[1]
                right = left + crop_width * image.size[0]
                top, left, bottom, right = int(top), int(left), int(bottom
                    ), int(right)
            keep_idx = points_img[:, 0] >= top
            keep_idx = np.logical_and(keep_idx, points_img[:, 0] < bottom)
            keep_idx = np.logical_and(keep_idx, points_img[:, 1] >= left)
            keep_idx = np.logical_and(keep_idx, points_img[:, 1] < right)
            if np.sum(keep_idx) > 100:
                valid_crop = True
                break
        if valid_crop:
            image = image.crop((left, top, right, bottom))
            points_img = points_img[keep_idx]
            points_img[:, 0] -= top
            points_img[:, 1] -= left
            masks_sam = masks_sam[top:bottom, left:right]
            masks_seem = [mask[top:bottom, left:right] for mask in masks_seem]
            points = points[keep_idx]
            if seg_label is not None:
                seg_label = seg_label[keep_idx]
            if len(self.rand_crop) > 0:
                points_img[:, 0] = float(self.crop_size[1]) / image.size[1
                    ] * np.floor(points_img[:, 0])
                points_img[:, 1] = float(self.crop_size[0]) / image.size[0
                    ] * np.floor(points_img[:, 1])
                image = image.resize(self.crop_size, Image.BILINEAR)
        else:
            print('No valid crop found for image', data_dict['camera_path'])
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
    if not self.is_train and seg_label is not None:
        out_dict['seg_label'] = torch.from_numpy(seg_label[idxs])
    out_dict['img_indices'] = out_dict['img_indices'][idxs]
    sampled_sam_mask, sampled_sam_indices = sample_mix_masks(masks_sam,
        out_dict['img_indices'])
    cut_mix_mask, cut_mix_indices = get_cut_masks(out_dict['img'], out_dict
        ['img_indices'])
    if self.pselab_data is not None:
        out_dict['pseudo_label_2d'] = torch.from_numpy(self.pselab_data[
            index]['pseudo_label_2d'][keep_idx][idxs])
        out_dict['sam_mix_pseudo_label_2d'] = out_dict['pseudo_label_2d'
            ].clone()
        out_dict['cut_mix_pseudo_label_2d'] = out_dict['pseudo_label_2d'
            ].clone()
        out_dict['sam_pseudo_label_2d'] = out_dict['pseudo_label_2d'].clone()[
            sampled_sam_indices]
        out_dict['cut_pseudo_label_2d'] = out_dict['pseudo_label_2d'].clone()[
            cut_mix_indices]
        out_dict['sam_mix_indices'] = out_dict['img_indices']
        out_dict['cut_mix_indices'] = out_dict['img_indices']
        if self.pselab_data[index]['pseudo_label_3d'] is None:
            out_dict['pseudo_label_3d'] = None
            out_dict['sam_mix_pseudo_label_3d'] = None
            out_dict['cut_mix_pseudo_label_3d'] = None
        else:
            out_dict['pseudo_label_3d'] = torch.from_numpy(self.pselab_data
                [index]['pseudo_label_3d'][keep_idx][idxs])
            out_dict['sam_mix_pseudo_label_3d'] = out_dict['pseudo_label_3d'
                ].clone()
            out_dict['cut_mix_pseudo_label_3d'] = out_dict['pseudo_label_3d'
                ].clone()
            out_dict['sam_pseudo_label_3d'] = out_dict['pseudo_label_3d'
                ].clone()[sampled_sam_indices]
            out_dict['cut_pseudo_label_3d'] = out_dict['pseudo_label_3d'
                ].clone()[cut_mix_indices]
    if self.output_orig:
        out_dict.update({'orig_seg_label': seg_label, 'orig_points_idx': idxs})
    """
        Prepare Mixed Data
        """
    out_dict['masks_seem'] = torch.from_numpy(np.stack(masks_seem))
    out_dict['sampled_sam_mask'] = torch.from_numpy(sampled_sam_mask)
    out_dict['sampled_sam_sel_indices'] = sampled_sam_indices
    out_dict['sam_mix_image'] = out_dict['img'].clone().permute(1, 2, 0)
    out_dict['sam_mix_coords'] = out_dict['coords'].clone()
    out_dict['sam_mix_feats'] = out_dict['feats'].clone()
    out_dict['cut_mask'] = torch.from_numpy(cut_mix_mask)
    out_dict['cut_sel_indices'] = cut_mix_indices
    out_dict['cut_mix_image'] = out_dict['img'].clone().permute(1, 2, 0)
    out_dict['cut_mix_coords'] = out_dict['coords'].clone()
    out_dict['cut_mix_feats'] = out_dict['feats'].clone()
    out_dict['img'] = out_dict['img'].permute(1, 2, 0)
    return out_dict
