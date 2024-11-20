def register_attn_hooks(self, unet):
    """Registering hooks to obtain the attention maps for guidance"""
    attn_procs = {}
    for name in unet.attn_processors.keys():
        if name.endswith('attn1.processor') or name.endswith(
            'fuser.attn.processor'):
            attn_procs[name] = unet.attn_processors[name]
            continue
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
        attn_procs[name] = AttnProcessorWithHook(attn_processor_key=name,
            hidden_size=hidden_size, cross_attention_dim=
            cross_attention_dim, hook=self.attn_hook, fast_attn=True,
            enabled=False)
    unet.set_attn_processor(attn_procs)
