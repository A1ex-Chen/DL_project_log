def get_peft_kwargs(rank_dict, network_alpha_dict, peft_state_dict, is_unet
    =True):
    rank_pattern = {}
    alpha_pattern = {}
    r = lora_alpha = list(rank_dict.values())[0]
    if len(set(rank_dict.values())) > 1:
        r = collections.Counter(rank_dict.values()).most_common()[0][0]
        rank_pattern = dict(filter(lambda x: x[1] != r, rank_dict.items()))
        rank_pattern = {k.split('.lora_B.')[0]: v for k, v in rank_pattern.
            items()}
    if network_alpha_dict is not None and len(network_alpha_dict) > 0:
        if len(set(network_alpha_dict.values())) > 1:
            lora_alpha = collections.Counter(network_alpha_dict.values()
                ).most_common()[0][0]
            alpha_pattern = dict(filter(lambda x: x[1] != lora_alpha,
                network_alpha_dict.items()))
            if is_unet:
                alpha_pattern = {'.'.join(k.split('.lora_A.')[0].split('.')
                    ).replace('.alpha', ''): v for k, v in alpha_pattern.
                    items()}
            else:
                alpha_pattern = {'.'.join(k.split('.down.')[0].split('.')[:
                    -1]): v for k, v in alpha_pattern.items()}
        else:
            lora_alpha = set(network_alpha_dict.values()).pop()
    target_modules = list({name.split('.lora')[0] for name in
        peft_state_dict.keys()})
    use_dora = any('lora_magnitude_vector' in k for k in peft_state_dict)
    lora_config_kwargs = {'r': r, 'lora_alpha': lora_alpha, 'rank_pattern':
        rank_pattern, 'alpha_pattern': alpha_pattern, 'target_modules':
        target_modules, 'use_dora': use_dora}
    return lora_config_kwargs
