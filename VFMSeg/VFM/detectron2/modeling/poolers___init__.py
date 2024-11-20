def __init__(self, output_size, scales, sampling_ratio, pooler_type,
    canonical_box_size=224, canonical_level=4):
    """
        Args:
            output_size (int, tuple[int] or list[int]): output size of the pooled region,
                e.g., 14 x 14. If tuple or list is given, the length must be 2.
            scales (list[float]): The scale for each low-level pooling op relative to
                the input image. For a feature map with stride s relative to the input
                image, scale is defined as 1/s. The stride must be power of 2.
                When there are multiple scales, they must form a pyramid, i.e. they must be
                a monotically decreasing geometric sequence with a factor of 1/2.
            sampling_ratio (int): The `sampling_ratio` parameter for the ROIAlign op.
            pooler_type (string): Name of the type of pooling operation that should be applied.
                For instance, "ROIPool" or "ROIAlignV2".
            canonical_box_size (int): A canonical box size in pixels (sqrt(box area)). The default
                is heuristically defined as 224 pixels in the FPN paper (based on ImageNet
                pre-training).
            canonical_level (int): The feature map level index from which a canonically-sized box
                should be placed. The default is defined as level 4 (stride=16) in the FPN paper,
                i.e., a box of size 224x224 will be placed on the feature with stride=16.
                The box placement for all boxes will be determined from their sizes w.r.t
                canonical_box_size. For example, a box whose area is 4x that of a canonical box
                should be used to pool features from feature level ``canonical_level+1``.

                Note that the actual input feature maps given to this module may not have
                sufficiently many levels for the input boxes. If the boxes are too large or too
                small for the input feature maps, the closest level will be used.
        """
    super().__init__()
    if isinstance(output_size, int):
        output_size = output_size, output_size
    assert len(output_size) == 2
    assert isinstance(output_size[0], int) and isinstance(output_size[1], int)
    self.output_size = output_size
    if pooler_type == 'ROIAlign':
        self.level_poolers = nn.ModuleList(ROIAlign(output_size,
            spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=
            False) for scale in scales)
    elif pooler_type == 'ROIAlignV2':
        self.level_poolers = nn.ModuleList(ROIAlign(output_size,
            spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=
            True) for scale in scales)
    elif pooler_type == 'ROIPool':
        self.level_poolers = nn.ModuleList(RoIPool(output_size,
            spatial_scale=scale) for scale in scales)
    elif pooler_type == 'ROIAlignRotated':
        self.level_poolers = nn.ModuleList(ROIAlignRotated(output_size,
            spatial_scale=scale, sampling_ratio=sampling_ratio) for scale in
            scales)
    else:
        raise ValueError('Unknown pooler type: {}'.format(pooler_type))
    min_level = -math.log2(scales[0])
    max_level = -math.log2(scales[-1])
    assert math.isclose(min_level, int(min_level)) and math.isclose(max_level,
        int(max_level)), 'Featuremap stride is not power of 2!'
    self.min_level = int(min_level)
    self.max_level = int(max_level)
    assert len(scales
        ) == self.max_level - self.min_level + 1, '[ROIPooler] Sizes of input featuremaps do not form a pyramid!'
    assert 0 <= self.min_level and self.min_level <= self.max_level
    self.canonical_level = canonical_level
    assert canonical_box_size > 0
    self.canonical_box_size = canonical_box_size
