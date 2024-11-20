def test_simple_inference_with_text_unet_block_scale_for_all_dict_options(self
    ):
    """Tests that any valid combination of lora block scales can be used in pipe.set_adapter"""

    def updown_options(blocks_with_tf, layers_per_block, value):
        """
            Generate every possible combination for how a lora weight dict for the up/down part can be.
            E.g. 2, {"block_1": 2}, {"block_1": [2,2,2]}, {"block_1": 2, "block_2": [2,2,2]}, ...
            """
        num_val = value
        list_val = [value] * layers_per_block
        node_opts = [None, num_val, list_val]
        node_opts_foreach_block = [node_opts] * len(blocks_with_tf)
        updown_opts = [num_val]
        for nodes in product(*node_opts_foreach_block):
            if all(n is None for n in nodes):
                continue
            opt = {}
            for b, n in zip(blocks_with_tf, nodes):
                if n is not None:
                    opt['block_' + str(b)] = n
            updown_opts.append(opt)
        return updown_opts

    def all_possible_dict_opts(unet, value):
        """
            Generate every possible combination for how a lora weight dict can be.
            E.g. 2, {"unet: {"down": 2}}, {"unet: {"down": [2,2,2]}}, {"unet: {"mid": 2, "up": [2,2,2]}}, ...
            """
        down_blocks_with_tf = [i for i, d in enumerate(unet.down_blocks) if
            hasattr(d, 'attentions')]
        up_blocks_with_tf = [i for i, u in enumerate(unet.up_blocks) if
            hasattr(u, 'attentions')]
        layers_per_block = unet.config.layers_per_block
        text_encoder_opts = [None, value]
        text_encoder_2_opts = [None, value]
        mid_opts = [None, value]
        down_opts = [None] + updown_options(down_blocks_with_tf,
            layers_per_block, value)
        up_opts = [None] + updown_options(up_blocks_with_tf, 
            layers_per_block + 1, value)
        opts = []
        for t1, t2, d, m, u in product(text_encoder_opts,
            text_encoder_2_opts, down_opts, mid_opts, up_opts):
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
    components, text_lora_config, unet_lora_config = self.get_dummy_components(
        self.scheduler_cls)
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    _, _, inputs = self.get_dummy_inputs(with_generator=False)
    pipe.text_encoder.add_adapter(text_lora_config, 'adapter-1')
    pipe.unet.add_adapter(unet_lora_config, 'adapter-1')
    if self.has_two_text_encoders:
        pipe.text_encoder_2.add_adapter(text_lora_config, 'adapter-1')
    for scale_dict in all_possible_dict_opts(pipe.unet, value=1234):
        if not self.has_two_text_encoders and 'text_encoder_2' in scale_dict:
            del scale_dict['text_encoder_2']
        pipe.set_adapters('adapter-1', scale_dict)
