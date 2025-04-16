def _convert_openai_clip(state_dict: Dict[str, torch.Tensor], model:
    VisionTransformer, prefix: str='visual.') ->Dict[str, torch.Tensor]:
    out_dict = {}
    swaps = [('conv1', 'patch_embed.proj'), ('positional_embedding',
        'pos_embed'), ('transformer.resblocks.', 'blocks.'), ('ln_pre',
        'norm_pre'), ('ln_post', 'norm'), ('ln_', 'norm'), ('in_proj_',
        'qkv.'), ('out_proj', 'proj'), ('mlp.c_fc', 'mlp.fc1'), (
        'mlp.c_proj', 'mlp.fc2')]
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        k = k.replace(prefix, '')
        for sp in swaps:
            k = k.replace(sp[0], sp[1])
        if k == 'proj':
            k = 'head.weight'
            v = v.transpose(0, 1)
            out_dict['head.bias'] = torch.zeros(v.shape[0])
        elif k == 'class_embedding':
            k = 'cls_token'
            v = v.unsqueeze(0).unsqueeze(1)
        elif k == 'pos_embed':
            v = v.unsqueeze(0)
            if v.shape[1] != model.pos_embed.shape[1]:
                v = resize_pos_embed(v, model.pos_embed, 0 if getattr(model,
                    'no_embed_class') else getattr(model,
                    'num_prefix_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict
