@staticmethod
def backward(ctx, grad_loss):
    logits, max_log_sum_exp, labels, smoothing, padding_idx = ctx.saved_tensors
    if not grad_loss.is_contiguous():
        grad_loss = grad_loss.contiguous()
    grad_loss.masked_fill_(labels == padding_idx.item(), 0)
    grad_logits = xentropy_cuda.backward(grad_loss.contiguous(), logits,
        max_log_sum_exp, labels, smoothing.item())
    return grad_logits, None, None, None, None
