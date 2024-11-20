def rename_state_dict_key(k):
    if k == 'embeddings.weight':
        return 'shared.weight'
    for parlai_name, hf_name in PATTERNS:
        k = k.replace(parlai_name, hf_name)
    if k.startswith('encoder'):
        k = k.replace('.attn', '.self_attn')
        k = k.replace('norm1', 'self_attn_layer_norm')
        k = k.replace('norm2', 'final_layer_norm')
    elif k.startswith('decoder'):
        k = k.replace('norm1', 'self_attn_layer_norm')
        k = k.replace('norm2', 'encoder_attn_layer_norm')
        k = k.replace('norm3', 'final_layer_norm')
    return k
