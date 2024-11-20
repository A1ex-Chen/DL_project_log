def get_normalized_probs(self, net_output, log_probs, sample=None):
    """Get normalized probabilities (or log probs) from a net's output."""
    logits = net_output[0].float()
    if log_probs:
        return F.log_softmax(logits, dim=-1)
    else:
        return F.softmax(logits, dim=-1)
