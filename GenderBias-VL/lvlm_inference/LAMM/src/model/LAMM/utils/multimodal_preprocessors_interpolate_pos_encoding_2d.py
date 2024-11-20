def interpolate_pos_encoding_2d(target_spatial_size, pos_embed):
    N = pos_embed.shape[1]
    if N == target_spatial_size:
        return pos_embed
    dim = pos_embed.shape[-1]
    pos_embed, updated = cast_if_src_dtype(pos_embed, torch.bfloat16, torch
        .float32)
    pos_embed = nn.functional.interpolate(pos_embed.reshape(1, int(math.
        sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2), scale_factor
        =math.sqrt(target_spatial_size / N), mode='bicubic')
    if updated:
        pos_embed, _ = cast_if_src_dtype(pos_embed, torch.float32, torch.
            bfloat16)
    pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return pos_embed
