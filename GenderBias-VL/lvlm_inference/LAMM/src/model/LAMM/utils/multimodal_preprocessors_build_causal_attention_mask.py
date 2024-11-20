def build_causal_attention_mask(context_length):
    mask = torch.empty(context_length, context_length, requires_grad=False)
    mask.fill_(float('-inf'))
    mask.triu_(1)
    return mask
