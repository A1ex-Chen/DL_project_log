def lock(self, unlocked_groups=0, freeze_bn_stats=False):
    """lock modules
        Args:
            unlocked_groups (int): leave last n layer groups unlocked (default: 0)
        """
    if not unlocked_groups:
        for param in self.trunk.parameters():
            param.requires_grad = False
        if freeze_bn_stats:
            freeze_batch_norm_2d(self.trunk)
    else:
        try:
            from timm.models.helpers import group_parameters, group_modules
        except ImportError:
            raise RuntimeError(
                'Please install latest timm `pip install git+https://github.com/rwightman/pytorch-image-models`'
                )
        matcher = self.trunk.group_matcher()
        gparams = group_parameters(self.trunk, matcher)
        max_layer_id = max(gparams.keys())
        max_layer_id = max_layer_id - unlocked_groups
        for group_idx in range(max_layer_id + 1):
            group = gparams[group_idx]
            for param in group:
                self.trunk.get_parameter(param).requires_grad = False
        if freeze_bn_stats:
            gmodules = group_modules(self.trunk, matcher, reverse=True)
            gmodules = {k for k, v in gmodules.items() if v <= max_layer_id}
            freeze_batch_norm_2d(self.trunk, gmodules)
