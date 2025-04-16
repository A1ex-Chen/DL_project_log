def tokenize_captions(captions):
    inputs = tokenizer(captions, max_length=tokenizer.model_max_length,
        padding='max_length', truncation=True, return_tensors='pt')
    return inputs.input_ids
