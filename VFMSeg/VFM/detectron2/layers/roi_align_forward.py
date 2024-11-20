def forward(self, input, rois):
    """
        Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N. The other 4 columns are xyxy.
        """
    assert rois.dim() == 2 and rois.size(1) == 5
    if input.is_quantized:
        input = input.dequantize()
    return roi_align(input, rois.to(dtype=input.dtype), self.output_size,
        self.spatial_scale, self.sampling_ratio, self.aligned)
