def build_criterion(vocab_size, padding_idx, smoothing):
    if smoothing == 0.0:
        loss_weight = torch.ones(vocab_size)
        loss_weight[padding_idx] = 0
        criterion = nn.CrossEntropyLoss(weight=loss_weight, size_average=False)
    else:
        criterion = LabelSmoothing(padding_idx, smoothing)
    return criterion
