@staticmethod
def convert_from_pytorch(pt_state: Dict, config: BertConfig) ->Dict:
    jax_state = dict(pt_state)
    for key, tensor in pt_state.items():
        key_parts = set(key.split('.'))
        if 'dense.weight' in key:
            del jax_state[key]
            key = key.replace('weight', 'kernel')
            jax_state[key] = tensor
        if {'query', 'key', 'value'} & key_parts:
            if 'bias' in key:
                jax_state[key] = tensor.reshape((config.num_attention_heads,
                    -1))
            elif 'weight':
                del jax_state[key]
                key = key.replace('weight', 'kernel')
                tensor = tensor.reshape((config.num_attention_heads, -1,
                    config.hidden_size)).transpose((2, 0, 1))
                jax_state[key] = tensor
        if 'attention.output.dense' in key:
            del jax_state[key]
            key = key.replace('attention.output.dense', 'attention.self.out')
            jax_state[key] = tensor
        if 'attention.output.LayerNorm' in key:
            del jax_state[key]
            key = key.replace('attention.output.LayerNorm',
                'attention.LayerNorm')
            jax_state[key] = tensor
        if 'intermediate.dense.kernel' in key or 'output.dense.kernel' in key:
            jax_state[key] = tensor.T
        if 'out.kernel' in key:
            jax_state[key] = tensor.reshape((config.hidden_size, config.
                num_attention_heads, -1)).transpose(1, 2, 0)
        if 'pooler.dense.kernel' in key:
            jax_state[key] = tensor.T
        if 'LayerNorm' in key:
            del jax_state[key]
            new_key = key.replace('LayerNorm', 'layer_norm')
            if 'weight' in key:
                new_key = new_key.replace('weight', 'gamma')
            elif 'bias' in key:
                new_key = new_key.replace('bias', 'beta')
            jax_state[new_key] = tensor
    return jax_state
