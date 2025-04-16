def convert_motion_module(original_state_dict):
    converted_state_dict = {}
    for k, v in original_state_dict.items():
        if 'pos_encoder' in k:
            continue
        else:
            converted_state_dict[k.replace('.norms.0', '.norm1').replace(
                '.norms.1', '.norm2').replace('.ff_norm', '.norm3').replace
                ('.attention_blocks.0', '.attn1').replace(
                '.attention_blocks.1', '.attn2').replace(
                '.temporal_transformer', '')] = v
    return converted_state_dict
