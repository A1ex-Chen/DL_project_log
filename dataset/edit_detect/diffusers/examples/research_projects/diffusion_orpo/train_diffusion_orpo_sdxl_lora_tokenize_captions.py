def tokenize_captions(tokenizers, examples):
    captions = []
    for caption in examples['caption']:
        captions.append(caption)
    tokens_one = tokenizers[0](captions, truncation=True, padding=
        'max_length', max_length=tokenizers[0].model_max_length,
        return_tensors='pt').input_ids
    tokens_two = tokenizers[1](captions, truncation=True, padding=
        'max_length', max_length=tokenizers[1].model_max_length,
        return_tensors='pt').input_ids
    return tokens_one, tokens_two
