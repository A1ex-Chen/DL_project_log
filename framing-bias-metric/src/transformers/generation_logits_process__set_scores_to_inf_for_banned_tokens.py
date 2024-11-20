def _set_scores_to_inf_for_banned_tokens(self, scores: torch.Tensor,
    banned_tokens: List[List[int]]) ->None:
    """
        Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be a
        list of list of banned tokens to ban in the format [[batch index, vocabulary position],...

        Args:
            scores: logits distribution of shape (batch size, vocabulary size)
            banned_tokens: list of list of tokens to ban of length (batch_size)
        """
    banned_mask_list = []
    for idx, batch_banned_tokens in enumerate(banned_tokens):
        for token in batch_banned_tokens:
            banned_mask_list.append([idx, token])
    if not banned_mask_list:
        return scores
    banned_mask = torch.LongTensor(banned_mask_list)
    indices = torch.ones(len(banned_mask))
    banned_mask = torch.sparse.LongTensor(banned_mask.t(), indices, scores.
        size()).to(scores.device).to_dense().bool()
    scores = scores.masked_fill(banned_mask, -float('inf'))
    return scores
