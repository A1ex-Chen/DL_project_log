def tokenize_prompt(tokenizer, prompt):
    return tokenizer(prompt, truncation=True, padding='max_length',
        max_length=77, return_tensors='pt').input_ids
