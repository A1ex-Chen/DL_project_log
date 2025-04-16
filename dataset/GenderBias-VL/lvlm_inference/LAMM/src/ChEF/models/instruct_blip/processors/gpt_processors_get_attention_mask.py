def get_attention_mask(self, seq):
    return torch.sum(seq != 1, dim=2) != 0
