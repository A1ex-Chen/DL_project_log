def shifted_scaled_dot_product_attention(self, attn: Attention, query:
    torch.Tensor, key: torch.Tensor, value: torch.Tensor) ->torch.Tensor:
    logits = torch.einsum('bhqd,bhkd->bhqk', query, key) * attn.scale
    logits[:, :, :, query.shape[2]:] += self.shared_score_shift
    probs = logits.softmax(-1)
    return torch.einsum('bhqk,bhkd->bhqd', probs, value)
