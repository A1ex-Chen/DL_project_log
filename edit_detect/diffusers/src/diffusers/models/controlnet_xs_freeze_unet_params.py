def freeze_unet_params(self) ->None:
    """Freeze the weights of the parts belonging to the base UNet2DConditionModel, and leave everything else unfrozen for fine
        tuning."""
    for param in self.parameters():
        param.requires_grad = True
    base_parts = ['base_time_proj', 'base_time_embedding',
        'base_add_time_proj', 'base_add_embedding', 'base_conv_in',
        'base_conv_norm_out', 'base_conv_act', 'base_conv_out']
    base_parts = [getattr(self, part) for part in base_parts if getattr(
        self, part) is not None]
    for part in base_parts:
        for param in part.parameters():
            param.requires_grad = False
    for d in self.down_blocks:
        d.freeze_base_params()
    self.mid_block.freeze_base_params()
    for u in self.up_blocks:
        u.freeze_base_params()
