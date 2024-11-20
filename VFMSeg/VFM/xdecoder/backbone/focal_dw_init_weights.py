def init_weights(self, pretrained=None):
    """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    if isinstance(pretrained, str):
        self.apply(_init_weights)
        logger = get_root_logger()
        load_checkpoint(self, pretrained, strict=False, logger=logger)
    elif pretrained is None:
        self.apply(_init_weights)
    else:
        raise TypeError('pretrained must be a str or None')
