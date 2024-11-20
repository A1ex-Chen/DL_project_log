def init_attn_processors(self, mask, token_idx, use_painta=True, use_rasg=
    True, painta_scale_factors=[2, 4], rasg_scale_factor=4,
    self_attention_layer_name='attn1', cross_attention_layer_name='attn2',
    list_of_painta_layer_names=None, list_of_rasg_layer_names=None):
    default_processor = AttnProcessor()
    width, height = mask.shape[-2:]
    width, height = (width // self.vae_scale_factor, height // self.
        vae_scale_factor)
    painta_scale_factors = [(x * self.vae_scale_factor) for x in
        painta_scale_factors]
    rasg_scale_factor = self.vae_scale_factor * rasg_scale_factor
    attn_processors = {}
    for x in self.unet.attn_processors:
        if (list_of_painta_layer_names is None and 
            self_attention_layer_name in x or list_of_painta_layer_names is not
            None and x in list_of_painta_layer_names):
            if use_painta:
                transformer_block = self.unet.get_submodule(x.replace(
                    '.attn1.processor', ''))
                attn_processors[x] = PAIntAAttnProcessor(transformer_block,
                    mask, token_idx, self.do_classifier_free_guidance,
                    painta_scale_factors)
            else:
                attn_processors[x] = default_processor
        elif list_of_rasg_layer_names is None and cross_attention_layer_name in x or list_of_rasg_layer_names is not None and x in list_of_rasg_layer_names:
            if use_rasg:
                attn_processors[x] = RASGAttnProcessor(mask, token_idx,
                    rasg_scale_factor)
            else:
                attn_processors[x] = default_processor
    self.unet.set_attn_processor(attn_processors)
