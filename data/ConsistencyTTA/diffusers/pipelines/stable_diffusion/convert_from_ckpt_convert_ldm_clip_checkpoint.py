def convert_ldm_clip_checkpoint(checkpoint):
    text_model = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
    keys = list(checkpoint.keys())
    text_model_dict = {}
    for key in keys:
        if key.startswith('cond_stage_model.transformer'):
            text_model_dict[key[len('cond_stage_model.transformer.'):]
                ] = checkpoint[key]
    text_model.load_state_dict(text_model_dict)
    return text_model
