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
    keys_to_rename = {}
    for k, v in new_state_dict.items():
        for weight_name in weights_to_convert:
            if f'mid.attn_1.{weight_name}.weight' in k:
                print(f'Reshaping {k} for SD format')
                new_state_dict[k] = reshape_weight_for_sd(v)
        for weight_name, real_weight_name in vae_extra_conversion_map:
            if (f'mid.attn_1.{weight_name}.weight' in k or 
                f'mid.attn_1.{weight_name}.bias' in k):
                keys_to_rename[k] = k.replace(weight_name, real_weight_name)
    for k, v in keys_to_rename.items():
        if k in new_state_dict:
            print(f'Renaming {k} to {v}')
            new_state_dict[v] = reshape_weight_for_sd(new_state_dict[k])
            del new_state_dict[k]
    return new_state_dict
