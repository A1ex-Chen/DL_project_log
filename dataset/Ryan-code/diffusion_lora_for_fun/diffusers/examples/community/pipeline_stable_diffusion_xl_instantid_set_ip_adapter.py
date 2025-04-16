def set_ip_adapter(self, model_ckpt, num_tokens, scale):
    unet = self.unet
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith('attn1.processor'
            ) else unet.config.cross_attention_dim
        if name.startswith('mid_block'):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith('up_blocks'):
            block_id = int(name[len('up_blocks.')])
            hidden_size = list(reversed(unet.config.block_out_channels))[
                block_id]
        elif name.startswith('down_blocks'):
            block_id = int(name[len('down_blocks.')])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor().to(unet.device, dtype=unet.dtype
                )
        else:
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim, scale=scale,
                num_tokens=num_tokens).to(unet.device, dtype=unet.dtype)
    unet.set_attn_processor(attn_procs)
    state_dict = torch.load(model_ckpt, map_location='cpu')
    ip_layers = torch.nn.ModuleList(self.unet.attn_processors.values())
    if 'ip_adapter' in state_dict:
        state_dict = state_dict['ip_adapter']
    ip_layers.load_state_dict(state_dict)
