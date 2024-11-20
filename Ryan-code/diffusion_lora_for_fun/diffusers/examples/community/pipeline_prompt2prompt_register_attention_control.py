def register_attention_control(self, controller):
    attn_procs = {}
    cross_att_count = 0
    for name in self.unet.attn_processors.keys():
        None if name.endswith('attn1.processor'
            ) else self.unet.config.cross_attention_dim
        if name.startswith('mid_block'):
            self.unet.config.block_out_channels[-1]
            place_in_unet = 'mid'
        elif name.startswith('up_blocks'):
            block_id = int(name[len('up_blocks.')])
            list(reversed(self.unet.config.block_out_channels))[block_id]
            place_in_unet = 'up'
        elif name.startswith('down_blocks'):
            block_id = int(name[len('down_blocks.')])
            self.unet.config.block_out_channels[block_id]
            place_in_unet = 'down'
        else:
            continue
        cross_att_count += 1
        attn_procs[name] = P2PCrossAttnProcessor(controller=controller,
            place_in_unet=place_in_unet)
    self.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count
