def _enable_shared_attention_processors(self, share_attention: bool,
    adain_queries: bool, adain_keys: bool, adain_values: bool,
    full_attention_share: bool, shared_score_scale: float,
    shared_score_shift: float, only_self_level: float):
    """Helper method to enable usage of Shared Attention Processor."""
    attn_procs = {}
    num_self_layers = len([name for name in self.unet.attn_processors.keys(
        ) if 'attn1' in name])
    only_self_vec = get_switch_vec(num_self_layers, only_self_level)
    for i, name in enumerate(self.unet.attn_processors.keys()):
        is_self_attention = 'attn1' in name
        if is_self_attention:
            if only_self_vec[i // 2]:
                attn_procs[name] = AttnProcessor2_0()
            else:
                attn_procs[name] = SharedAttentionProcessor(share_attention
                    =share_attention, adain_queries=adain_queries,
                    adain_keys=adain_keys, adain_values=adain_values,
                    full_attention_share=full_attention_share,
                    shared_score_scale=shared_score_scale,
                    shared_score_shift=shared_score_shift)
        else:
            attn_procs[name] = AttnProcessor2_0()
    self.unet.set_attn_processor(attn_procs)
