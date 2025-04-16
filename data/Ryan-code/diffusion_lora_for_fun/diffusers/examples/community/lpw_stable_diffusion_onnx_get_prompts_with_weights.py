def get_prompts_with_weights(pipe, prompt: List[str], max_length: int):
    """
    Tokenize a list of prompts and return its tokens with weights of each token.

    No padding, starting or ending token is included.
    """
    tokens = []
    weights = []
    truncated = False
    for text in prompt:
        texts_and_weights = parse_prompt_attention(text)
        text_token = []
        text_weight = []
        for word, weight in texts_and_weights:
            token = pipe.tokenizer(word, return_tensors='np').input_ids[0, 1:-1
                ]
            text_token += list(token)
            text_weight += [weight] * len(token)
            if len(text_token) > max_length:
                truncated = True
                break
        if len(text_token) > max_length:
            truncated = True
            text_token = text_token[:max_length]
            text_weight = text_weight[:max_length]
        tokens.append(text_token)
        weights.append(text_weight)
    if truncated:
        logger.warning(
            'Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples'
            )
    return tokens, weights
