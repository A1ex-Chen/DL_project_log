def tokenize_captions(examples, is_train=True):
    captions = []
    for caption in examples[caption_column]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f'Caption column `{caption_column}` should contain either strings or lists of strings.'
                )
    inputs = tokenizer(captions, max_length=tokenizer.model_max_length,
        padding='max_length', truncation=True, return_tensors='pt')
    text_input_ids = inputs.input_ids
    text_mask = inputs.attention_mask.bool()
    return text_input_ids, text_mask
