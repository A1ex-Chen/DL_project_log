@staticmethod
@once_differentiable
def backward(ctx, grad_output):
    (value, value_spatial_shapes, value_level_start_index,
        sampling_locations, attention_weights) = ctx.saved_tensors
    grad_value, grad_sampling_loc, grad_attn_weight = (MSDA.
        ms_deform_attn_backward(value, value_spatial_shapes,
        value_level_start_index, sampling_locations, attention_weights,
        grad_output, ctx.im2col_step))
    return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None
