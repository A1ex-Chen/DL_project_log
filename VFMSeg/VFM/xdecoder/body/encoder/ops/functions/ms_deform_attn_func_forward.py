@staticmethod
def forward(ctx, value, value_spatial_shapes, value_level_start_index,
    sampling_locations, attention_weights, im2col_step):
    ctx.im2col_step = im2col_step
    output = MSDA.ms_deform_attn_forward(value, value_spatial_shapes,
        value_level_start_index, sampling_locations, attention_weights, ctx
        .im2col_step)
    ctx.save_for_backward(value, value_spatial_shapes,
        value_level_start_index, sampling_locations, attention_weights)
    return output
