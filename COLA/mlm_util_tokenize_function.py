def tokenize_function(examples):
    inputs, targets = [], []
    for i in range(len(examples['e0'])):
        text = examples['e0'][i] + ' ' + tokenizer.mask_token + ' ' + examples[
            'e1'][i]
        inputs.append(text)
        targets.append(token2id_mapping[examples['label'][i]])
    model_inputs = tokenizer(inputs, padding=padding, truncation=True,
        max_length=max_seq_length)
    model_inputs['labels'] = targets
    return model_inputs
