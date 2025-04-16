def convert_open_clip_checkpoint(checkpoint):
    text_model = CLIPTextModel.from_pretrained('stabilityai/stable-diffusion-2'
        , subfolder='text_encoder')
    keys = list(checkpoint.keys())
    text_model_dict = {}
    if 'cond_stage_model.model.text_projection' in checkpoint:
        d_model = int(checkpoint['cond_stage_model.model.text_projection'].
            shape[0])
    else:
        d_model = 1024
    text_model_dict['text_model.embeddings.position_ids'
        ] = text_model.text_model.embeddings.get_buffer('position_ids')
    for key in keys:
        if 'resblocks.23' in key:
            continue
        if key in textenc_conversion_map:
            text_model_dict[textenc_conversion_map[key]] = checkpoint[key]
        if key.startswith('cond_stage_model.model.transformer.'):
            new_key = key[len('cond_stage_model.model.transformer.'):]
            if new_key.endswith('.in_proj_weight'):
                new_key = new_key[:-len('.in_proj_weight')]
                new_key = textenc_pattern.sub(lambda m: protected[re.escape
                    (m.group(0))], new_key)
                text_model_dict[new_key + '.q_proj.weight'] = checkpoint[key][
                    :d_model, :]
                text_model_dict[new_key + '.k_proj.weight'] = checkpoint[key][
                    d_model:d_model * 2, :]
                text_model_dict[new_key + '.v_proj.weight'] = checkpoint[key][
                    d_model * 2:, :]
            elif new_key.endswith('.in_proj_bias'):
                new_key = new_key[:-len('.in_proj_bias')]
                new_key = textenc_pattern.sub(lambda m: protected[re.escape
                    (m.group(0))], new_key)
                text_model_dict[new_key + '.q_proj.bias'] = checkpoint[key][:
                    d_model]
                text_model_dict[new_key + '.k_proj.bias'] = checkpoint[key][
                    d_model:d_model * 2]
                text_model_dict[new_key + '.v_proj.bias'] = checkpoint[key][
                    d_model * 2:]
            else:
                new_key = textenc_pattern.sub(lambda m: protected[re.escape
                    (m.group(0))], new_key)
                text_model_dict[new_key] = checkpoint[key]
    text_model.load_state_dict(text_model_dict)
    return text_model
