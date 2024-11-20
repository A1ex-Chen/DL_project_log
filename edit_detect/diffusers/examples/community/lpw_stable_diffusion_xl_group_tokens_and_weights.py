def group_tokens_and_weights(token_ids: list, weights: list, pad_last_block
    =False):
    """
    Produce tokens and weights in groups and pad the missing tokens

    Args:
        token_ids (list)
            The token ids from tokenizer
        weights (list)
            The weights list from function get_prompts_tokens_with_weights
        pad_last_block (bool)
            Control if fill the last token list to 75 tokens with eos
    Returns:
        new_token_ids (2d list)
        new_weights (2d list)

    Example:
        token_groups,weight_groups = group_tokens_and_weights(
            token_ids = token_id_list
            , weights = token_weight_list
        )
    """
    bos, eos = 49406, 49407
    new_token_ids = []
    new_weights = []
    while len(token_ids) >= 75:
        head_75_tokens = [token_ids.pop(0) for _ in range(75)]
        head_75_weights = [weights.pop(0) for _ in range(75)]
        temp_77_token_ids = [bos] + head_75_tokens + [eos]
        temp_77_weights = [1.0] + head_75_weights + [1.0]
        new_token_ids.append(temp_77_token_ids)
        new_weights.append(temp_77_weights)
    if len(token_ids) > 0:
        padding_len = 75 - len(token_ids) if pad_last_block else 0
        temp_77_token_ids = [bos] + token_ids + [eos] * padding_len + [eos]
        new_token_ids.append(temp_77_token_ids)
        temp_77_weights = [1.0] + weights + [1.0] * padding_len + [1.0]
        new_weights.append(temp_77_weights)
    return new_token_ids, new_weights
