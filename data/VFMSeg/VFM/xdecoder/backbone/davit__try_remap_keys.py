def _try_remap_keys(self, pretrained_dict):
    remap_keys = {'conv_embeds': 'convs', 'main_blocks': 'blocks',
        '0.cpe.0.proj': 'spatial_block.conv1.fn.dw', '0.attn':
        'spatial_block.window_attn.fn', '0.cpe.1.proj':
        'spatial_block.conv2.fn.dw', '0.mlp': 'spatial_block.ffn.fn.net',
        '1.cpe.0.proj': 'channel_block.conv1.fn.dw', '1.attn':
        'channel_block.channel_attn.fn', '1.cpe.1.proj':
        'channel_block.conv2.fn.dw', '1.mlp': 'channel_block.ffn.fn.net',
        '0.norm1': 'spatial_block.window_attn.norm', '0.norm2':
        'spatial_block.ffn.norm', '1.norm1':
        'channel_block.channel_attn.norm', '1.norm2': 'channel_block.ffn.norm'}
    full_key_mappings = {}
    for k in pretrained_dict.keys():
        old_k = k
        for remap_key in remap_keys.keys():
            if remap_key in k:
                print(f'=> Repace {remap_key} with {remap_keys[remap_key]}')
                k = k.replace(remap_key, remap_keys[remap_key])
        full_key_mappings[old_k] = k
    return full_key_mappings
