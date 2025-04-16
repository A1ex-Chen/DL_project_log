def _get_pos_embedding(npatch_per_img, pos_embed, patches_layout,
    input_shape, first_patch_idx=1):
    pos_embed = interpolate_pos_encoding(npatch_per_img, pos_embed,
        patches_layout, input_shape=input_shape, first_patch_idx=
        first_patch_idx)
    return pos_embed
