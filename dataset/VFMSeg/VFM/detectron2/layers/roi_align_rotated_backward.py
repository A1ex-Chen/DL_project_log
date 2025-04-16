@staticmethod
@once_differentiable
def backward(ctx, grad_output):
    rois, = ctx.saved_tensors
    output_size = ctx.output_size
    spatial_scale = ctx.spatial_scale
    sampling_ratio = ctx.sampling_ratio
    bs, ch, h, w = ctx.input_shape
    grad_input = torch.ops.detectron2.roi_align_rotated_backward(grad_output,
        rois, spatial_scale, output_size[0], output_size[1], bs, ch, h, w,
        sampling_ratio)
    return grad_input, None, None, None, None, None
