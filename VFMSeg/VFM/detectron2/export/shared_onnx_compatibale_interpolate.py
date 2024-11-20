def onnx_compatibale_interpolate(input, size=None, scale_factor=None, mode=
    'nearest', align_corners=None):
    if size is None and scale_factor is not None:
        if input.dim() == 4:
            if isinstance(scale_factor, (int, float)):
                height_scale, width_scale = scale_factor, scale_factor
            else:
                assert isinstance(scale_factor, (tuple, list))
                assert len(scale_factor) == 2
                height_scale, width_scale = scale_factor
            assert not align_corners, 'No matching C2 op for align_corners == True'
            if mode == 'nearest':
                return torch.ops._caffe2.ResizeNearest(input, order='NCHW',
                    width_scale=width_scale, height_scale=height_scale)
            elif mode == 'bilinear':
                logger.warning(
                    "Use F.conv_transpose2d for bilinear interpolate because there's no such C2 op, this may cause significant slowdown and the boundary pixels won't be as same as using F.interpolate due to padding."
                    )
                assert height_scale == width_scale
                return BilinearInterpolation(input, up_scale=height_scale)
        logger.warning(
            'Output size is not static, it might cause ONNX conversion issue')
    return interp(input, size, scale_factor, mode, align_corners)
