def prepare_unet(self, attention_store, PnP: bool=False):
    attn_procs = {}
    for name in self.unet.attn_processors.keys():
        if name.startswith('mid_block'):
            place_in_unet = 'mid'
        elif name.startswith('up_blocks'):
            place_in_unet = 'up'
        elif name.startswith('down_blocks'):
            place_in_unet = 'down'
        else:
            continue
        if 'attn2' in name and place_in_unet != 'mid':
            attn_procs[name] = LEDITSCrossAttnProcessor(attention_store=
                attention_store, place_in_unet=place_in_unet, pnp=PnP,
                editing_prompts=self.enabled_editing_prompts)
        else:
            attn_procs[name] = AttnProcessor()
    self.unet.set_attn_processor(attn_procs)
