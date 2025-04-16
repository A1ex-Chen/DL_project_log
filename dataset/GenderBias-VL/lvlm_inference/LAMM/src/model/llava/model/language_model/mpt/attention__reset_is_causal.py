def _reset_is_causal(num_query_tokens: int, num_key_tokens: int,
    original_is_causal: bool):
    if original_is_causal and num_query_tokens != num_key_tokens:
        if num_query_tokens != 1:
            raise NotImplementedError(
                'MPT does not support query and key with different number of tokens, unless number of query tokens is 1.'
                )
        else:
            return False
    return original_is_causal
