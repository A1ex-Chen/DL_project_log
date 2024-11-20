@configurable
def __init__(self, *, sizes, aspect_ratios, strides, angles, offset=0.5):
    """
        This interface is experimental.

        Args:
            sizes (list[list[float]] or list[float]):
                If sizes is list[list[float]], sizes[i] is the list of anchor sizes
                (i.e. sqrt of anchor area) to use for the i-th feature map.
                If sizes is list[float], the sizes are used for all feature maps.
                Anchor sizes are given in absolute lengths in units of
                the input image; they do not dynamically scale if the input image size changes.
            aspect_ratios (list[list[float]] or list[float]): list of aspect ratios
                (i.e. height / width) to use for anchors. Same "broadcast" rule for `sizes` applies.
            strides (list[int]): stride of each input feature.
            angles (list[list[float]] or list[float]): list of angles (in degrees CCW)
                to use for anchors. Same "broadcast" rule for `sizes` applies.
            offset (float): Relative offset between the center of the first anchor and the top-left
                corner of the image. Value has to be in [0, 1).
                Recommend to use 0.5, which means half stride.
        """
    super().__init__()
    self.strides = strides
    self.num_features = len(self.strides)
    sizes = _broadcast_params(sizes, self.num_features, 'sizes')
    aspect_ratios = _broadcast_params(aspect_ratios, self.num_features,
        'aspect_ratios')
    angles = _broadcast_params(angles, self.num_features, 'angles')
    self.cell_anchors = self._calculate_anchors(sizes, aspect_ratios, angles)
    self.offset = offset
    assert 0.0 <= self.offset < 1.0, self.offset
