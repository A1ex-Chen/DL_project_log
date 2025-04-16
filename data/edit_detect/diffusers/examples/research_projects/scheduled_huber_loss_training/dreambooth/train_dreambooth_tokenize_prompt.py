def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length
    text_inputs = tokenizer(prompt, truncation=True, padding='max_length',
        max_length=max_length, return_tensors='pt')
    return text_inputs
