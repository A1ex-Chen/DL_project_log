@classmethod
def from_unet2d(cls, unet: UNet2DConditionModel, motion_adapter: Optional[
    MotionAdapter]=None, load_weights: bool=True):
    has_motion_adapter = motion_adapter is not None
    if has_motion_adapter:
        motion_adapter.to(device=unet.device)
    config = dict(unet.config)
    config['_class_name'] = cls.__name__
    down_blocks = []
    for down_blocks_type in config['down_block_types']:
        if 'CrossAttn' in down_blocks_type:
            down_blocks.append('CrossAttnDownBlockMotion')
        else:
            down_blocks.append('DownBlockMotion')
    config['down_block_types'] = down_blocks
    up_blocks = []
    for down_blocks_type in config['up_block_types']:
        if 'CrossAttn' in down_blocks_type:
            up_blocks.append('CrossAttnUpBlockMotion')
        else:
            up_blocks.append('UpBlockMotion')
    config['up_block_types'] = up_blocks
    if has_motion_adapter:
        config['motion_num_attention_heads'] = motion_adapter.config[
            'motion_num_attention_heads']
        config['motion_max_seq_length'] = motion_adapter.config[
            'motion_max_seq_length']
        config['use_motion_mid_block'] = motion_adapter.config[
            'use_motion_mid_block']
        if motion_adapter.config['conv_in_channels']:
            config['in_channels'] = motion_adapter.config['conv_in_channels']
    if not config.get('num_attention_heads'):
        config['num_attention_heads'] = config['attention_head_dim']
    config = FrozenDict(config)
    model = cls.from_config(config)
    if not load_weights:
        return model
    if has_motion_adapter and motion_adapter.config['conv_in_channels']:
        model.conv_in = motion_adapter.conv_in
        updated_conv_in_weight = torch.cat([unet.conv_in.weight,
            motion_adapter.conv_in.weight[:, 4:, :, :]], dim=1)
        model.conv_in.load_state_dict({'weight': updated_conv_in_weight,
            'bias': unet.conv_in.bias})
    else:
        model.conv_in.load_state_dict(unet.conv_in.state_dict())
    model.time_proj.load_state_dict(unet.time_proj.state_dict())
    model.time_embedding.load_state_dict(unet.time_embedding.state_dict())
    for i, down_block in enumerate(unet.down_blocks):
        model.down_blocks[i].resnets.load_state_dict(down_block.resnets.
            state_dict())
        if hasattr(model.down_blocks[i], 'attentions'):
            model.down_blocks[i].attentions.load_state_dict(down_block.
                attentions.state_dict())
        if model.down_blocks[i].downsamplers:
            model.down_blocks[i].downsamplers.load_state_dict(down_block.
                downsamplers.state_dict())
    for i, up_block in enumerate(unet.up_blocks):
        model.up_blocks[i].resnets.load_state_dict(up_block.resnets.
            state_dict())
        if hasattr(model.up_blocks[i], 'attentions'):
            model.up_blocks[i].attentions.load_state_dict(up_block.
                attentions.state_dict())
        if model.up_blocks[i].upsamplers:
            model.up_blocks[i].upsamplers.load_state_dict(up_block.
                upsamplers.state_dict())
    model.mid_block.resnets.load_state_dict(unet.mid_block.resnets.state_dict()
        )
    model.mid_block.attentions.load_state_dict(unet.mid_block.attentions.
        state_dict())
    if unet.conv_norm_out is not None:
        model.conv_norm_out.load_state_dict(unet.conv_norm_out.state_dict())
    if unet.conv_act is not None:
        model.conv_act.load_state_dict(unet.conv_act.state_dict())
    model.conv_out.load_state_dict(unet.conv_out.state_dict())
    if has_motion_adapter:
        model.load_motion_modules(motion_adapter)
    model.to(unet.dtype)
    return model
