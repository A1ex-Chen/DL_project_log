def _softmax_cross_entropy_with_logits(logits, labels):
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.permute(*transpose_param)
    loss_ftor = nn.CrossEntropyLoss(reduce=False)
    loss = loss_ftor(logits, labels.max(dim=-1)[1])
    return loss
