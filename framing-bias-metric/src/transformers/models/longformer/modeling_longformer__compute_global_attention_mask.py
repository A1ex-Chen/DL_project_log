def _compute_global_attention_mask(input_ids, sep_token_id,
    before_sep_token=True):
    """
    Computes global attention mask by putting attention on all tokens before `sep_token_id` if `before_sep_token is
    True` else after `sep_token_id`.
    """
    question_end_index = _get_question_end_index(input_ids, sep_token_id)
    question_end_index = question_end_index.unsqueeze(dim=1)
    attention_mask = torch.arange(input_ids.shape[1], device=input_ids.device)
    if before_sep_token is True:
        attention_mask = (attention_mask.expand_as(input_ids) <
            question_end_index).to(torch.uint8)
    else:
        attention_mask = (attention_mask.expand_as(input_ids) > 
            question_end_index + 1).to(torch.uint8) * (attention_mask.
            expand_as(input_ids) < input_ids.shape[-1]).to(torch.uint8)
    return attention_mask
