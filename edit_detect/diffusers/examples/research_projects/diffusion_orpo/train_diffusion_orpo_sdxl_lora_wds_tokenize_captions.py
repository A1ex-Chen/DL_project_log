def tokenize_captions(tokenizers, sample):
    tokens_one = tokenizers[0](sample['original_prompt'], truncation=True,
        padding='max_length', max_length=tokenizers[0].model_max_length,
        return_tensors='pt').input_ids
    tokens_two = tokenizers[1](sample['original_prompt'], truncation=True,
        padding='max_length', max_length=tokenizers[1].model_max_length,
        return_tensors='pt').input_ids
    return tokens_one, tokens_two
