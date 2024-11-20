def register_attention_control(self):
    attn_procs = {}
    cross_att_count = 0
    for name in self.unet.attn_processors.keys():
        if name.startswith('mid_block'):
            place_in_unet = 'mid'
        elif name.startswith('up_blocks'):
            place_in_unet = 'up'
        elif name.startswith('down_blocks'):
            place_in_unet = 'down'
        else:
            continue
        cross_att_count += 1
        attn_procs[name] = AttendExciteAttnProcessor(attnstore=self.
            attention_store, place_in_unet=place_in_unet)
    self.unet.set_attn_processor(attn_procs)
    self.attention_store.num_att_layers = cross_att_count
