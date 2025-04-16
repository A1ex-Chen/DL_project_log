def forward(self, input, rois):
    """
        Args:
            input: NCHW images
            rois: Bx6 boxes. First column is the index into N.
                The other 5 columns are (x_ctr, y_ctr, width, height, angle_degrees).
        """
    assert rois.dim() == 2 and rois.size(1) == 6
    orig_dtype = input.dtype
    if orig_dtype == torch.float16:
        input = input.float()
        rois = rois.float()
    return roi_align_rotated(input, rois, self.output_size, self.
        spatial_scale, self.sampling_ratio).to(dtype=orig_dtype)
