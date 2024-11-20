def get_nll(self, seq_logits, doc_scores, target, reduce_loss=False,
    epsilon=0.0, n_docs=None):
    n_docs = n_docs if n_docs is not None else self.config.n_docs
    target = torch.cat([target[:, 1:], target.new(target.shape[0], 1).fill_
        (self.config.generator.pad_token_id)], 1)

    def _mask_pads(ll, smooth_obj):
        pad_mask = target.eq(self.config.generator.pad_token_id)
        if pad_mask.any():
            ll.masked_fill_(pad_mask, 0.0)
            smooth_obj.masked_fill_(pad_mask, 0.0)
        return ll.squeeze(-1), smooth_obj.squeeze(-1)
    rag_logprobs = self.marginalize(seq_logits, doc_scores, n_docs)
    target = target.unsqueeze(-1)
    assert target.dim() == rag_logprobs.dim()
    ll = rag_logprobs.gather(dim=-1, index=target)
    smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)
    ll, smooth_obj = _mask_pads(ll, smooth_obj)
    ll = ll.sum(1)
    smooth_obj = smooth_obj.sum(1)
    nll_loss = -ll
    smooth_loss = -smooth_obj
    if reduce_loss:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / rag_logprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss
