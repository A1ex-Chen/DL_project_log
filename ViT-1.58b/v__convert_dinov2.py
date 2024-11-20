def _convert_dinov2(state_dict: Dict[str, torch.Tensor], model:
    VisionTransformer) ->Dict[str, torch.Tensor]:
    import re
    out_dict = {}
    state_dict.pop('mask_token', None)
    if 'register_tokens' in state_dict:
        out_dict['reg_token'] = state_dict.pop('register_tokens')
        out_dict['cls_token'] = state_dict.pop('cls_token') + state_dict[
            'pos_embed'][:, 0]
        out_dict['pos_embed'] = state_dict.pop('pos_embed')[:, 1:]
    for k, v in state_dict.items():
        if re.match('blocks\\.(\\d+)\\.mlp\\.w12\\.(?:weight|bias)', k):
            out_dict[k.replace('w12', 'fc1')] = v
            continue
        elif re.match('blocks\\.(\\d+)\\.mlp\\.w3\\.(?:weight|bias)', k):
            out_dict[k.replace('w3', 'fc2')] = v
            continue
        out_dict[k] = v
    return out_dict
