def convert_open_clip_checkpoint(checkpoint):
    checkpoint = checkpoint['text_encoder']
    text_model = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
    keys = list(checkpoint.keys())
    text_model_dict = {}
    if 'cond_stage_model.model.text_projection' in checkpoint:
        d_model = int(checkpoint['cond_stage_model.model.text_projection'].
            shape[0])
    else:
        d_model = 1024
    for key in keys:
        if 'resblocks.23' in key:
            continue
        if key in textenc_conversion_map:
            text_model_dict[textenc_conversion_map[key]] = checkpoint[key]
        new_key = key[len('transformer.'):]
        if new_key.endswith('.in_proj_weight'):
            new_key = new_key[:-len('.in_proj_weight')]
            new_key = textenc_pattern.sub(lambda m: protected[re.escape(m.
                group(0))], new_key)
            text_model_dict[new_key + '.q_proj.weight'] = checkpoint[key][:
                d_model, :]
            text_model_dict[new_key + '.k_proj.weight'] = checkpoint[key][
                d_model:d_model * 2, :]
            text_model_dict[new_key + '.v_proj.weight'] = checkpoint[key][
                d_model * 2:, :]
        elif new_key.endswith('.in_proj_bias'):
            new_key = new_key[:-len('.in_proj_bias')]
            new_key = textenc_pattern.sub(lambda m: protected[re.escape(m.
                group(0))], new_key)
            text_model_dict[new_key + '.q_proj.bias'] = checkpoint[key][:
                d_model]
            text_model_dict[new_key + '.k_proj.bias'] = checkpoint[key][d_model
                :d_model * 2]
            text_model_dict[new_key + '.v_proj.bias'] = checkpoint[key][
                d_model * 2:]
        else:
            if key != 'transformer.text_model.embeddings.position_ids':
                new_key = textenc_pattern.sub(lambda m: protected[re.escape
                    (m.group(0))], new_key)
                text_model_dict[new_key] = checkpoint[key]
            if (key ==
                'transformer.text_model.embeddings.token_embedding.weight'):
                text_model_dict['text_model.embeddings.token_embedding.weight'
                    ] = checkpoint[key]
    text_model_dict.pop(
        'text_model.embeddings.transformer.text_model.embeddings.token_embedding.weight'
        )
    text_model.load_state_dict(text_model_dict)
    return text_model
