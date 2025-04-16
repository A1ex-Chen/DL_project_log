def get_attention_scores(self, query, key, attention_mask=None):
    dtype = query.dtype
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
    attention_probs = attention_scores.softmax(dim=-1)
    del attention_scores
    attention_probs = attention_probs.to(dtype)
    return attention_probs
