def convert_stable_cascade_unet_single_file_to_diffusers(checkpoint, **kwargs):
    is_stage_c = 'clip_txt_mapper.weight' in checkpoint
    if is_stage_c:
        state_dict = {}
        for key in checkpoint.keys():
            if key.endswith('in_proj_weight'):
                weights = checkpoint[key].chunk(3, 0)
                state_dict[key.replace('attn.in_proj_weight', 'to_q.weight')
                    ] = weights[0]
                state_dict[key.replace('attn.in_proj_weight', 'to_k.weight')
                    ] = weights[1]
                state_dict[key.replace('attn.in_proj_weight', 'to_v.weight')
                    ] = weights[2]
            elif key.endswith('in_proj_bias'):
                weights = checkpoint[key].chunk(3, 0)
                state_dict[key.replace('attn.in_proj_bias', 'to_q.bias')
                    ] = weights[0]
                state_dict[key.replace('attn.in_proj_bias', 'to_k.bias')
                    ] = weights[1]
                state_dict[key.replace('attn.in_proj_bias', 'to_v.bias')
                    ] = weights[2]
            elif key.endswith('out_proj.weight'):
                weights = checkpoint[key]
                state_dict[key.replace('attn.out_proj.weight',
                    'to_out.0.weight')] = weights
            elif key.endswith('out_proj.bias'):
                weights = checkpoint[key]
                state_dict[key.replace('attn.out_proj.bias', 'to_out.0.bias')
                    ] = weights
            else:
                state_dict[key] = checkpoint[key]
    else:
        state_dict = {}
        for key in checkpoint.keys():
            if key.endswith('in_proj_weight'):
                weights = checkpoint[key].chunk(3, 0)
                state_dict[key.replace('attn.in_proj_weight', 'to_q.weight')
                    ] = weights[0]
                state_dict[key.replace('attn.in_proj_weight', 'to_k.weight')
                    ] = weights[1]
                state_dict[key.replace('attn.in_proj_weight', 'to_v.weight')
                    ] = weights[2]
            elif key.endswith('in_proj_bias'):
                weights = checkpoint[key].chunk(3, 0)
                state_dict[key.replace('attn.in_proj_bias', 'to_q.bias')
                    ] = weights[0]
                state_dict[key.replace('attn.in_proj_bias', 'to_k.bias')
                    ] = weights[1]
                state_dict[key.replace('attn.in_proj_bias', 'to_v.bias')
                    ] = weights[2]
            elif key.endswith('out_proj.weight'):
                weights = checkpoint[key]
                state_dict[key.replace('attn.out_proj.weight',
                    'to_out.0.weight')] = weights
            elif key.endswith('out_proj.bias'):
                weights = checkpoint[key]
                state_dict[key.replace('attn.out_proj.bias', 'to_out.0.bias')
                    ] = weights
            elif key.endswith('clip_mapper.weight'):
                weights = checkpoint[key]
                state_dict[key.replace('clip_mapper.weight',
                    'clip_txt_pooled_mapper.weight')] = weights
            elif key.endswith('clip_mapper.bias'):
                weights = checkpoint[key]
                state_dict[key.replace('clip_mapper.bias',
                    'clip_txt_pooled_mapper.bias')] = weights
            else:
                state_dict[key] = checkpoint[key]
    return state_dict
