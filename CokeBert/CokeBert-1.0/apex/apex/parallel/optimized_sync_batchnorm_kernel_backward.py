@staticmethod
def backward(ctx, grad_output):
    grad_output = grad_output.contiguous()
    torch.cuda.nvtx.range_push('sync_BN_bw')
    saved_input, weight, mean, inv_std = ctx.saved_tensors
    process_group = ctx.process_group
    channel_last = ctx.channel_last
    world_size = ctx.world_size
    grad_input = grad_weight = grad_bias = None
    if channel_last:
        mean_dy, mean_dy_xmu, grad_weight, grad_bias = syncbn.reduce_bn_c_last(
            grad_output, saved_input, mean, inv_std, weight)
    else:
        mean_dy, mean_dy_xmu, grad_weight, grad_bias = syncbn.reduce_bn(
            grad_output, saved_input, mean, inv_std, weight)
    if ctx.needs_input_grad[0]:
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(mean_dy, ReduceOp.SUM, process_group)
            mean_dy = mean_dy / world_size
            torch.distributed.all_reduce(mean_dy_xmu, ReduceOp.SUM,
                process_group)
            mean_dy_xmu = mean_dy_xmu / world_size
        if channel_last:
            grad_input = syncbn.batchnorm_backward_c_last(grad_output,
                saved_input, mean, inv_std, weight, mean_dy, mean_dy_xmu)
        else:
            grad_input = syncbn.batchnorm_backward(grad_output, saved_input,
                mean, inv_std, weight, mean_dy, mean_dy_xmu)
    if weight is None or not ctx.needs_input_grad[1]:
        grad_weight = None
    if weight is None or not ctx.needs_input_grad[2]:
        grad_bias = None
    torch.cuda.nvtx.range_pop()
    return (grad_input, grad_weight, grad_bias, None, None, None, None,
        None, None, None)
