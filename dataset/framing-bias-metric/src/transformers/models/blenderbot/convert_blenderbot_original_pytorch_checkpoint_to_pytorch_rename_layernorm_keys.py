def rename_layernorm_keys(sd):
    keys = ['model.encoder.layernorm_embedding.weight',
        'model.encoder.layernorm_embedding.bias',
        'model.decoder.layernorm_embedding.weight',
        'model.decoder.layernorm_embedding.bias']
    for k in keys:
        v = sd.pop(k)
        new_k = k.replace('layernorm_embedding', 'layer_norm')
        assert new_k not in sd
        sd[new_k] = v
