def label_smoothing_opt_1(x, target, padding_idx, smoothing):
    logprobs = torch.nn.functional.log_softmax(x, dim=-1, dtype=torch.float32)
    pad_mask = target == padding_idx
    ll_loss = logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
    smooth_loss = logprobs.mean(dim=-1)
    loss = (smoothing - 1.0) * ll_loss - smoothing * smooth_loss
    loss.masked_fill_(pad_mask, 0)
    return loss
