def tokenize_captions(tokenizer, examples):
    max_length = tokenizer.model_max_length
    captions = []
    for caption in examples['caption']:
        captions.append(caption)
    text_inputs = tokenizer(captions, truncation=True, padding='max_length',
        max_length=max_length, return_tensors='pt')
    return text_inputs.input_ids
