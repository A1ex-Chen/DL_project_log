def interpolate_pos_encoding(npatch_per_img, pos_embed, patches_layout,
    input_shape=None, first_patch_idx=1):
    assert first_patch_idx == 0 or first_patch_idx == 1, 'there is 1 CLS token or none'
    N = pos_embed.shape[1] - first_patch_idx
    if npatch_per_img == N:
        return pos_embed
    assert patches_layout[-1] == patches_layout[-2
        ], 'Interpolation of pos embed not supported for non-square layouts'
    class_emb = pos_embed[:, :first_patch_idx]
    pos_embed = pos_embed[:, first_patch_idx:]
    if input_shape is None or patches_layout[0] == 1:
        pos_embed = interpolate_pos_encoding_2d(npatch_per_img, pos_embed)
    elif patches_layout[0] > 1:
        assert len(input_shape) == 4, 'temporal interpolation not supported'
        num_frames = patches_layout[0]
        num_spatial_tokens = patches_layout[1] * patches_layout[2]
        pos_embed = pos_embed.view(1, num_frames, num_spatial_tokens, -1)
        pos_embed = interpolate_pos_encoding_2d(npatch_per_img, pos_embed[0,
            0, ...].unsqueeze(0))
    else:
        raise ValueError("This type of interpolation isn't implemented")
    return torch.cat((class_emb, pos_embed), dim=1)
