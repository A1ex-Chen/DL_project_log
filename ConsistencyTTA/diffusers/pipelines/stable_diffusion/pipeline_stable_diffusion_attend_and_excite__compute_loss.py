@staticmethod
def _compute_loss(max_attention_per_index: List[torch.Tensor]) ->torch.Tensor:
    """Computes the attend-and-excite loss using the maximum attention value for each token."""
    losses = [max(0, 1.0 - curr_max) for curr_max in max_attention_per_index]
    loss = max(losses)
    return loss
