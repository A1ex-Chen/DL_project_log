def all_possible_dict_opts(unet, value):
    """
            Generate every possible combination for how a lora weight dict can be.
            E.g. 2, {"unet: {"down": 2}}, {"unet: {"down": [2,2,2]}}, {"unet: {"mid": 2, "up": [2,2,2]}}, ...
            """
    down_blocks_with_tf = [i for i, d in enumerate(unet.down_blocks) if
        hasattr(d, 'attentions')]
    up_blocks_with_tf = [i for i, u in enumerate(unet.up_blocks) if hasattr
        (u, 'attentions')]
    layers_per_block = unet.config.layers_per_block
    text_encoder_opts = [None, value]
    text_encoder_2_opts = [None, value]
    mid_opts = [None, value]
    down_opts = [None] + updown_options(down_blocks_with_tf,
        layers_per_block, value)
    up_opts = [None] + updown_options(up_blocks_with_tf, layers_per_block +
        1, value)
    opts = []
    for t1, t2, d, m, u in product(text_encoder_opts, text_encoder_2_opts,
        down_opts, mid_opts, up_opts):
        if all(o is None for o in (t1, t2, d, m, u)):
            continue
        opt = {}
        if t1 is not None:
            opt['text_encoder'] = t1
        if t2 is not None:
            opt['text_encoder_2'] = t2
        if all(o is None for o in (d, m, u)):
            continue
        opt['unet'] = {}
        if d is not None:
            opt['unet']['down'] = d
        if m is not None:
            opt['unet']['mid'] = m
        if u is not None:
            opt['unet']['up'] = u
        opts.append(opt)
    return opts
