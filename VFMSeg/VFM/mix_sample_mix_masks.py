def sample_mix_masks(sam_mask_data, indices):
    """
    SAM Mix
    """
    sampled_ids_list = sample(list(discard(set(sam_mask_data.flatten()))))
    sampled_2d_masks = sam_mask_data == sampled_ids_list[0]
    for id in range(1, len(sampled_ids_list)):
        sampled_2d_masks = sampled_2d_masks | (sam_mask_data ==
            sampled_ids_list[id])
    sampled_2d_to_3d_indices = sampled_2d_masks[indices[:, 0], indices[:, 1]
        ] == True
    return sampled_2d_masks, sampled_2d_to_3d_indices
