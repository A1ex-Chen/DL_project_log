def _get_question_end_index(input_ids, sep_token_id):
    """
    Computes the index of the first occurance of `sep_token_id`.
    """
    sep_token_indices = (input_ids == sep_token_id).nonzero()
    batch_size = input_ids.shape[0]
    assert sep_token_indices.shape[1
        ] == 2, '`input_ids` should have two dimensions'
    assert sep_token_indices.shape[0
        ] == 3 * batch_size, f'There should be exactly three separator tokens: {sep_token_id} in every sample for questions answering. You might also consider to set `global_attention_mask` manually in the forward function to avoid this error.'
    return sep_token_indices.view(batch_size, 3, 2)[:, 0, 1]
