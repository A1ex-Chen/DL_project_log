@staticmethod
def forward(ctx, input, weight, bias, running_mean, running_variance, eps,
    process_group, world_size):
    torch.cuda.nvtx.range_push('sync_BN_fw')
    c_last_input = input.transpose(1, -1).contiguous().clone()
    ctx.save_for_backward(c_last_input, weight, bias, running_mean,
        running_variance)
    ctx.eps = eps
    ctx.process_group = process_group
    ctx.world_size = world_size
    c_last_input = (c_last_input - running_mean) / torch.sqrt(
        running_variance + eps)
    if weight is not None:
        c_last_input = c_last_input * weight
    if bias is not None:
        c_last_input = c_last_input + bias
    torch.cuda.nvtx.range_pop()
    return c_last_input.transpose(1, -1).contiguous().clone()
