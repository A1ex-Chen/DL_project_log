def text_encoder():
    print('loading CLIP text encoder')
    clip_name = 'openai/clip-vit-large-patch14'
    pad_token = '!'
    tokenizer_model = CLIPTokenizer.from_pretrained(clip_name, pad_token=
        pad_token, device_map='auto')
    assert tokenizer_model.convert_tokens_to_ids(pad_token) == 0
    text_encoder_model = CLIPTextModelWithProjection.from_pretrained(clip_name)
    print('done loading CLIP text encoder')
    return text_encoder_model, tokenizer_model
