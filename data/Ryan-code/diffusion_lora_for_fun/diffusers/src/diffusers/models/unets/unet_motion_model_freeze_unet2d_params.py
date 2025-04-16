def freeze_unet2d_params(self) ->None:
    """Freeze the weights of just the UNet2DConditionModel, and leave the motion modules
        unfrozen for fine tuning.
        """
    for param in self.parameters():
        param.requires_grad = False
    for down_block in self.down_blocks:
        motion_modules = down_block.motion_modules
        for param in motion_modules.parameters():
            param.requires_grad = True
    for up_block in self.up_blocks:
        motion_modules = up_block.motion_modules
        for param in motion_modules.parameters():
            param.requires_grad = True
    if hasattr(self.mid_block, 'motion_modules'):
        motion_modules = self.mid_block.motion_modules
        for param in motion_modules.parameters():
            param.requires_grad = True
