def marginalize(self, seq_logits, doc_scores, n_docs=None):
    n_docs = n_docs if n_docs is not None else self.config.n_docs
    seq_logprobs = torch.nn.functional.log_softmax(seq_logits, dim=-1).view(
        seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1))
    doc_logprobs = torch.log_softmax(doc_scores, dim=1)
    log_prob_sum = seq_logprobs + doc_logprobs.unsqueeze(-1).unsqueeze(-1)
    return torch.logsumexp(log_prob_sum, dim=1)
