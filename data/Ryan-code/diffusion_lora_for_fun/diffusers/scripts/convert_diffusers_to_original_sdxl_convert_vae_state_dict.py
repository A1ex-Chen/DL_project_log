def convert_vae_state_dict(vae_state_dict):
    mapping = {k: k for k in vae_state_dict.keys()}
    for k, v in mapping.items():
        for sd_part, hf_part in vae_conversion_map:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    for k, v in mapping.items():
        if 'attentions' in k:
            for sd_part, hf_part in vae_conversion_map_attn:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    new_state_dict = {v: vae_state_dict[k] for k, v in mapping.items()}
    weights_to_convert = ['q', 'k', 'v', 'proj_out']
    for k, v in new_state_dict.items():
        for weight_name in weights_to_convert:
            if f'mid.attn_1.{weight_name}.weight' in k:
                print(f'Reshaping {k} for SD format')
                new_state_dict[k] = reshape_weight_for_sd(v)
    return new_state_dict
