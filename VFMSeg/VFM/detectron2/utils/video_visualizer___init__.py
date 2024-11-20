def __init__(self, metadata, instance_mode=ColorMode.IMAGE):
    """
        Args:
            metadata (MetadataCatalog): image metadata.
        """
    self.metadata = metadata
    self._old_instances = []
    assert instance_mode in [ColorMode.IMAGE, ColorMode.IMAGE_BW
        ], 'Other mode not supported yet.'
    self._instance_mode = instance_mode
    self._max_num_instances = self.metadata.get('max_num_instances', 74)
    self._assigned_colors = {}
    self._color_pool = random_colors(self._max_num_instances, rgb=True,
        maximum=1)
    self._color_idx_set = set(range(len(self._color_pool)))
