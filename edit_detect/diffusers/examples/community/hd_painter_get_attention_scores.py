def get_attention_scores(self, query: torch.Tensor, key: torch.Tensor,
    attention_mask: torch.Tensor=None) ->torch.Tensor:
    """
    Compute the attention scores.

    Args:
        query (`torch.Tensor`): The query tensor.
        key (`torch.Tensor`): The key tensor.
        attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

    Returns:
        `torch.Tensor`: The attention probabilities/scores.
    """
    if self.upcast_attention:
        query = query.float()
        key = key.float()
    if attention_mask is None:
        baddbmm_input = torch.empty(query.shape[0], query.shape[1], key.
            shape[1], dtype=query.dtype, device=query.device)
        beta = 0
    else:
        baddbmm_input = attention_mask
        beta = 1
    attention_scores = torch.baddbmm(baddbmm_input, query, key.transpose(-1,
        -2), beta=beta, alpha=self.scale)
    del baddbmm_input
    if self.upcast_softmax:
        attention_scores = attention_scores.float()
    return attention_scores
