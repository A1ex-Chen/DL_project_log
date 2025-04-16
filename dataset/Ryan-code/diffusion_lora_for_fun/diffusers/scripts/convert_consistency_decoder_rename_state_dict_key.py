def rename_state_dict_key(k):
    k = k.replace('blocks.', '')
    for i in range(5):
        k = k.replace(f'down_{i}_', f'down.{i}.')
        k = k.replace(f'conv_{i}.', f'{i}.')
        k = k.replace(f'up_{i}_', f'up.{i}.')
        k = k.replace(f'mid_{i}', f'mid.{i}')
    k = k.replace('upsamp.', '4.')
    k = k.replace('downsamp.', '3.')
    k = k.replace('f_t.w', 'f_t.weight').replace('f_t.b', 'f_t.bias')
    k = k.replace('f_1.w', 'f_1.weight').replace('f_1.b', 'f_1.bias')
    k = k.replace('f_2.w', 'f_2.weight').replace('f_2.b', 'f_2.bias')
    k = k.replace('f_s.w', 'f_s.weight').replace('f_s.b', 'f_s.bias')
    k = k.replace('f.w', 'f.weight').replace('f.b', 'f.bias')
    k = k.replace('gn_1.g', 'gn_1.weight').replace('gn_1.b', 'gn_1.bias')
    k = k.replace('gn_2.g', 'gn_2.weight').replace('gn_2.b', 'gn_2.bias')
    k = k.replace('gn.g', 'gn.weight').replace('gn.b', 'gn.bias')
    return k
