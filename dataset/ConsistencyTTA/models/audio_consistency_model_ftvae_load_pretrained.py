def load_pretrained(self, state_dict: Mapping[str, Any], strict: bool=True):
    return_info = super().load_pretrained(state_dict, strict)
    for module, name in zip([self.vae.decoder, self.vae.post_quant_conv,
        self.ema_vae_decoder, self.ema_vae_pqconv], ['vae.decoder',
        'vae.post_quant_conv', 'ema_vae_decoder', 'ema_vae_pqconv']):
        name_offset = len(name) + 1
        module_sd = {}
        for key, val in state_dict.items():
            if name in key:
                if 'loss' in key:
                    new_key = key[5:]
                    if new_key[name_offset:] not in module_sd.keys():
                        module_sd[new_key[name_offset:]] = val
                else:
                    module_sd[key[name_offset:]] = val
        module.load_state_dict(module_sd)
    self.vae.ema_decoder = self.ema_vae_decoder
    self.vae.ema_post_quant_conv = self.ema_vae_pqconv
    return return_info
