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
    tokens_one = tokenize_prompt(tokenizer_one, captions)
    tokens_two = tokenize_prompt(tokenizer_two, captions)
    return tokens_one, tokens_two
