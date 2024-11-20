def multi_scale_deformable_attn_pytorch(value: torch.Tensor,
    value_spatial_shapes: torch.Tensor, sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor) ->torch.Tensor:
    """
    Multiscale deformable attention.

    https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = (sampling_locations
        .shape)
    value_list = value.split([(H_ * W_) for H_, W_ in value_spatial_shapes],
        dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs *
            num_heads, embed_dims, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2
            ).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode=
            'bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(bs *
        num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
        attention_weights).sum(-1).view(bs, num_heads * embed_dims, num_queries
        )
    return output.transpose(1, 2).contiguous()
