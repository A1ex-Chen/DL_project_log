@staticmethod
def backward(ctx, *args):
    if not torch.autograd._is_checkpoint_valid():
        raise RuntimeError(
            'Checkpointing is not compatible with .grad(), please use .backward() if possible'
            )
    inputs = ctx.saved_tensors
    rng_devices = []
    if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
        rng_devices = ctx.fwd_gpu_devices
    with torch.random.fork_rng(devices=rng_devices, enabled=ctx.
        preserve_rng_state):
        if ctx.preserve_rng_state:
            torch.set_rng_state(ctx.fwd_cpu_state)
            if ctx.had_cuda_in_fwd:
                set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
        detached_inputs = detach_variable(inputs)
        with torch.enable_grad(), torch.cuda.amp.autocast(ctx.
            had_autocast_in_fwd):
            outputs = ctx.run_function(*detached_inputs)
    if isinstance(outputs, torch.Tensor):
        outputs = outputs,
    torch.autograd.backward(outputs, args)
    grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for
        inp in detached_inputs)
    return (None, None) + grads
