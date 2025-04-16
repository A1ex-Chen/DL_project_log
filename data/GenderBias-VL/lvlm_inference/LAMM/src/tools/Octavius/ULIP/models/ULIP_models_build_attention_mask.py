def build_attention_mask(self):
    mask = torch.empty(self.context_length, self.context_length)
    mask.fill_(float('-inf'))
    mask.triu_(1)
    return mask
