@staticmethod
def forward(ctx, logits, labels, smoothing=0.0, padding_idx=0,
    half_to_float=False):
    losses, max_log_sum_exp = xentropy_cuda.forward(logits, labels,
        smoothing, half_to_float)
    losses.masked_fill_(labels == padding_idx, 0)
    ctx.save_for_backward(logits, max_log_sum_exp, labels, torch.
        FloatTensor([smoothing]), torch.LongTensor([padding_idx]))
    return losses
