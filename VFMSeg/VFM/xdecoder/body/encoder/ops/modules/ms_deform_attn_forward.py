def forward(self, query, reference_points, input_flatten,
    input_spatial_shapes, input_level_start_index, input_padding_mask=None):
    """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \\sum_{l=0}^{L-1} H_l \\cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \\sum_{l=0}^{L-1} H_l \\cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
    N, Len_q, _ = query.shape
    N, Len_in, _ = input_flatten.shape
    assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum(
        ) == Len_in
    value = self.value_proj(input_flatten)
    if input_padding_mask is not None:
        value = value.masked_fill(input_padding_mask[..., None], float(0))
    value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
    sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.
        n_heads, self.n_levels, self.n_points, 2)
    attention_weights = self.attention_weights(query).view(N, Len_q, self.
        n_heads, self.n_levels * self.n_points)
    attention_weights = F.softmax(attention_weights, -1).view(N, Len_q,
        self.n_heads, self.n_levels, self.n_points)
    if reference_points.shape[-1] == 2:
        offset_normalizer = torch.stack([input_spatial_shapes[..., 1],
            input_spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points[:, :, None, :, None, :
            ] + sampling_offsets / offset_normalizer[None, None, None, :,
            None, :]
    elif reference_points.shape[-1] == 4:
        sampling_locations = reference_points[:, :, None, :, None, :2
            ] + sampling_offsets / self.n_points * reference_points[:, :,
            None, :, None, 2:] * 0.5
    else:
        raise ValueError(
            'Last dim of reference_points must be 2 or 4, but get {} instead.'
            .format(reference_points.shape[-1]))
    try:
        output = MSDeformAttnFunction.apply(value, input_spatial_shapes,
            input_level_start_index, sampling_locations, attention_weights,
            self.im2col_step)
    except:
        output = ms_deform_attn_core_pytorch(value, input_spatial_shapes,
            sampling_locations, attention_weights)
    output = self.output_proj(output)
    return output
