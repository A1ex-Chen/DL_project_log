def _convert_deprecated_attention_blocks(self, state_dict: OrderedDict) ->None:
    deprecated_attention_block_paths = []

    def recursive_find_attn_block(name, module):
        if hasattr(module, '_from_deprecated_attn_block'
            ) and module._from_deprecated_attn_block:
            deprecated_attention_block_paths.append(name)
        for sub_name, sub_module in module.named_children():
            sub_name = sub_name if name == '' else f'{name}.{sub_name}'
            recursive_find_attn_block(sub_name, sub_module)
    recursive_find_attn_block('', self)
    for path in deprecated_attention_block_paths:
        if f'{path}.query.weight' in state_dict:
            state_dict[f'{path}.to_q.weight'] = state_dict.pop(
                f'{path}.query.weight')
        if f'{path}.query.bias' in state_dict:
            state_dict[f'{path}.to_q.bias'] = state_dict.pop(
                f'{path}.query.bias')
        if f'{path}.key.weight' in state_dict:
            state_dict[f'{path}.to_k.weight'] = state_dict.pop(
                f'{path}.key.weight')
        if f'{path}.key.bias' in state_dict:
            state_dict[f'{path}.to_k.bias'] = state_dict.pop(f'{path}.key.bias'
                )
        if f'{path}.value.weight' in state_dict:
            state_dict[f'{path}.to_v.weight'] = state_dict.pop(
                f'{path}.value.weight')
        if f'{path}.value.bias' in state_dict:
            state_dict[f'{path}.to_v.bias'] = state_dict.pop(
                f'{path}.value.bias')
        if f'{path}.proj_attn.weight' in state_dict:
            state_dict[f'{path}.to_out.0.weight'] = state_dict.pop(
                f'{path}.proj_attn.weight')
        if f'{path}.proj_attn.bias' in state_dict:
            state_dict[f'{path}.to_out.0.bias'] = state_dict.pop(
                f'{path}.proj_attn.bias')
