def get_preprocess_function(data_args, tokenizer, logger, token2id_mapping):
    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f'The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). Picking 1024 instead. You can change that default value by passing --max_seq_length xxx.'
                )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f'The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for themodel ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}.'
                )
        max_seq_length = min(data_args.max_seq_length, tokenizer.
            model_max_length)
    padding = 'max_length' if data_args.pad_to_max_length else False

    def tokenize_function(examples):
        inputs, targets = [], []
        for i in range(len(examples['e0'])):
            text = examples['e0'][i
                ] + ' ' + tokenizer.mask_token + ' ' + examples['e1'][i]
            inputs.append(text)
            targets.append(token2id_mapping[examples['label'][i]])
        model_inputs = tokenizer(inputs, padding=padding, truncation=True,
            max_length=max_seq_length)
        model_inputs['labels'] = targets
        return model_inputs
    return tokenize_function
