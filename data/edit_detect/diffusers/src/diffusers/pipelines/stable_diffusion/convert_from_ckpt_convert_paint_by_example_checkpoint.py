def convert_paint_by_example_checkpoint(checkpoint, local_files_only=False):
    config = CLIPVisionConfig.from_pretrained('openai/clip-vit-large-patch14',
        local_files_only=local_files_only)
    model = PaintByExampleImageEncoder(config)
    keys = list(checkpoint.keys())
    text_model_dict = {}
    for key in keys:
        if key.startswith('cond_stage_model.transformer'):
            text_model_dict[key[len('cond_stage_model.transformer.'):]
                ] = checkpoint[key]
    model.model.load_state_dict(text_model_dict)
    keys_mapper = {k[len('cond_stage_model.mapper.res'):]: v for k, v in
        checkpoint.items() if k.startswith('cond_stage_model.mapper')}
    MAPPING = {'attn.c_qkv': ['attn1.to_q', 'attn1.to_k', 'attn1.to_v'],
        'attn.c_proj': ['attn1.to_out.0'], 'ln_1': ['norm1'], 'ln_2': [
        'norm3'], 'mlp.c_fc': ['ff.net.0.proj'], 'mlp.c_proj': ['ff.net.2']}
    mapped_weights = {}
    for key, value in keys_mapper.items():
        prefix = key[:len('blocks.i')]
        suffix = key.split(prefix)[-1].split('.')[-1]
        name = key.split(prefix)[-1].split(suffix)[0][1:-1]
        mapped_names = MAPPING[name]
        num_splits = len(mapped_names)
        for i, mapped_name in enumerate(mapped_names):
            new_name = '.'.join([prefix, mapped_name, suffix])
            shape = value.shape[0] // num_splits
            mapped_weights[new_name] = value[i * shape:(i + 1) * shape]
    model.mapper.load_state_dict(mapped_weights)
    model.final_layer_norm.load_state_dict({'bias': checkpoint[
        'cond_stage_model.final_ln.bias'], 'weight': checkpoint[
        'cond_stage_model.final_ln.weight']})
    model.proj_out.load_state_dict({'bias': checkpoint['proj_out.bias'],
        'weight': checkpoint['proj_out.weight']})
    model.uncond_vector.data = torch.nn.Parameter(checkpoint[
        'learnable_vector'])
    return model
