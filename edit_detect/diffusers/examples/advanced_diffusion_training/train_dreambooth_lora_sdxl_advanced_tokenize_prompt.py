def tokenize_prompt(tokenizer, prompt, add_special_tokens=False):
    text_inputs = tokenizer(prompt, padding='max_length', max_length=
        tokenizer.model_max_length, truncation=True, add_special_tokens=
        add_special_tokens, return_tensors='pt')
    text_input_ids = text_inputs.input_ids
    return text_input_ids
