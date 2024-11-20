def collate_scn_base(input_dict_list, output_orig, with_vfm, output_image=True
    ):
    """
    Custom collate function for SCN. The batch size is always 1,
    but the batch indices are appended to the locations.
    :param input_dict_list: a list of dicts from the dataloader
    :param output_orig: whether to output original point cloud/labels/indices
    :param output_image: whether to output images
    :return: Collated data batch as dict
    """
    locs = []
    feats = []
    labels = []
    if output_image:
        imgs = []
        img_idxs = []
        if with_vfm:
            masks_seem = []
            sampled_sam_mask = []
            sampled_sam_indices = []
            sam_mix_indices = []
            sam_mix_image = []
            sam_mix_label_2d = []
            sam_label = []
            sam_mix_coords = []
            sam_mix_feats = []
            sam_mix_label_3d = []
            cut_mask = []
            cut_indices = []
            cut_mix_indices = []
            cut_mix_image = []
            cut_mix_label_2d = []
            cut_label = []
            cut_mix_coords = []
            cut_mix_feats = []
            cut_mix_label_3d = []
    if output_orig:
        orig_seg_label = []
        orig_points_idx = []
    output_pselab = 'pseudo_label_2d' in input_dict_list[0].keys()
    if output_pselab:
        pseudo_label_2d = []
        pseudo_label_3d = []
        sam_mix_pseudo_label_2d = []
        sam_mix_pseudo_label_3d = []
        sam_pseudo_label_2d = []
        sam_pseudo_label_3d = []
        cut_mix_pseudo_label_2d = []
        cut_mix_pseudo_label_3d = []
        cut_pseudo_label_2d = []
        cut_pseudo_label_3d = []
    sam_seg_labels = 'sam_mix_label_2d' in input_dict_list[0].keys()
    exist_sam_mix_indices = 'sam_mix_indices' in input_dict_list[0].keys()
    for idx, input_dict in enumerate(input_dict_list):
        coords = input_dict['coords']
        batch_idxs = torch.LongTensor(coords.shape[0], 1).fill_(idx)
        locs.append(torch.cat([coords, batch_idxs], 1))
        feats.append(input_dict['feats'])
        if 'seg_label' in input_dict.keys():
            labels.append(input_dict['seg_label'])
            if not output_pselab and sam_seg_labels:
                sam_mix_label_2d.append(input_dict['sam_mix_label_2d'])
                sam_mix_label_3d.append(input_dict['sam_mix_label_3d'])
                sam_label.append(input_dict['sam_label'])
                cut_mix_label_2d.append(input_dict['cut_mix_label_2d'])
                cut_mix_label_3d.append(input_dict['cut_mix_label_3d'])
                cut_label.append(input_dict['cut_label'])
        if output_image:
            imgs.append(input_dict['img'])
            img_idxs.append(input_dict['img_indices'])
            sam_mix_image.append(input_dict['sam_mix_image'])
            sampled_sam_indices.append(input_dict['sampled_sam_sel_indices'])
            cut_mix_image.append(input_dict['cut_mix_image'])
            cut_indices.append(input_dict['cut_sel_indices'])
            if exist_sam_mix_indices:
                sam_mix_indices.append(input_dict['sam_mix_indices'])
                cut_mix_indices.append(input_dict['cut_mix_indices'])
        if output_orig:
            orig_seg_label.append(input_dict['orig_seg_label'])
            orig_points_idx.append(input_dict['orig_points_idx'])
        if output_pselab:
            pseudo_label_2d.append(input_dict['pseudo_label_2d'])
            sam_mix_pseudo_label_2d.append(input_dict[
                'sam_mix_pseudo_label_2d'])
            cut_mix_pseudo_label_2d.append(input_dict[
                'cut_mix_pseudo_label_2d'])
            sam_pseudo_label_2d.append(input_dict['sam_pseudo_label_2d'])
            cut_pseudo_label_2d.append(input_dict['cut_pseudo_label_2d'])
            if input_dict['pseudo_label_3d'] is not None:
                pseudo_label_3d.append(input_dict['pseudo_label_3d'])
                sam_mix_pseudo_label_3d.append(input_dict[
                    'sam_mix_pseudo_label_3d'])
                cut_mix_pseudo_label_3d.append(input_dict[
                    'cut_mix_pseudo_label_3d'])
                sam_pseudo_label_3d.append(input_dict['sam_pseudo_label_3d'])
                cut_pseudo_label_3d.append(input_dict['cut_pseudo_label_3d'])
        if with_vfm:
            masks_seem.append(input_dict['masks_seem'])
            sampled_sam_mask.append(input_dict['sampled_sam_mask'])
            cut_mask.append(input_dict['cut_mask'])
            sam_mix_coords.append(torch.cat([input_dict['sam_mix_coords'],
                batch_idxs.clone()], 1))
            sam_mix_feats.append(input_dict['sam_mix_feats'])
            cut_mix_coords.append(torch.cat([input_dict['cut_mix_coords'],
                batch_idxs.clone()], 1))
            cut_mix_feats.append(input_dict['cut_mix_feats'])
    locs = torch.cat(locs, 0)
    feats = torch.cat(feats, 0)
    out_dict = {'x': [locs, feats]}
    if labels:
        labels = torch.cat(labels, 0)
        out_dict['seg_label'] = labels
        if not output_pselab and sam_seg_labels:
            out_dict['sam_mix_label_2d'] = sam_mix_label_2d
            out_dict['cut_mix_label_2d'] = cut_mix_label_2d
            out_dict['sam_mix_label_3d'] = sam_mix_label_3d
            out_dict['cut_mix_label_3d'] = cut_mix_label_3d
            out_dict['sam_label'] = sam_label
            out_dict['cut_label'] = cut_label
    if output_image:
        out_dict['img'] = torch.stack(imgs)
        out_dict['img_indices'] = img_idxs
        out_dict['sam_mix_image'] = torch.stack(sam_mix_image)
        out_dict['sampled_sam_sel_indices'] = sampled_sam_indices
        out_dict['cut_mix_image'] = torch.stack(cut_mix_image)
        out_dict['cut_sel_indices'] = cut_indices
        if exist_sam_mix_indices:
            out_dict['sam_mix_indices'] = sam_mix_indices
            out_dict['cut_mix_indices'] = cut_mix_indices
    if with_vfm:
        out_dict['sam_mix_x'] = [sam_mix_coords, sam_mix_feats]
        out_dict['cut_mix_x'] = [cut_mix_coords, cut_mix_feats]
        out_dict['masks_seem'] = torch.stack(masks_seem)
        out_dict['sampled_sam_mask'] = torch.stack(sampled_sam_mask)
        out_dict['cut_mask'] = torch.stack(cut_mask)
    if output_orig:
        out_dict['orig_seg_label'] = orig_seg_label
        out_dict['orig_points_idx'] = orig_points_idx
    if output_pselab:
        out_dict['pseudo_label_2d'] = torch.cat(pseudo_label_2d, 0)
        out_dict['pseudo_label_3d'] = torch.cat(pseudo_label_3d, 0
            ) if pseudo_label_3d else pseudo_label_3d
        out_dict['sam_mix_pseudo_label_2d'] = sam_mix_pseudo_label_2d
        out_dict['cut_mix_pseudo_label_2d'] = cut_mix_pseudo_label_2d
        out_dict['sam_pseudo_label_2d'] = sam_pseudo_label_2d
        out_dict['cut_pseudo_label_2d'] = cut_pseudo_label_2d
        out_dict['sam_mix_pseudo_label_3d'] = sam_mix_pseudo_label_3d
        out_dict['cut_mix_pseudo_label_3d'] = cut_mix_pseudo_label_3d
        out_dict['sam_pseudo_label_3d'] = sam_pseudo_label_3d
        out_dict['cut_pseudo_label_3d'] = cut_pseudo_label_3d
    return out_dict
