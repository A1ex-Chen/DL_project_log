def post_backward_with_master_weights(self, scaler):
    stash = self._amp_stash
    fp16_grads_needing_unscale = []
    new_fp32_grads = []
    fp16_grads_needing_unscale_with_stash = []
    preexisting_fp32_grads = []
    for fp16_param, fp32_param in zip(stash.all_fp16_params, stash.
        all_fp32_from_fp16_params):
        if fp16_param.grad is None and fp32_param.grad is not None:
            continue
        elif fp16_param.grad is not None and fp32_param.grad is None:
            fp32_param.grad = torch.empty_like(fp32_param)
            fp16_grads_needing_unscale.append(fp16_param.grad)
            new_fp32_grads.append(fp32_param.grad)
        elif fp16_param.grad is not None and fp32_param.grad is not None:
            fp16_grads_needing_unscale_with_stash.append(fp16_param.grad)
            preexisting_fp32_grads.append(fp32_param.grad)
        else:
            continue
    if len(fp16_grads_needing_unscale) > 0:
        scaler.unscale(fp16_grads_needing_unscale, new_fp32_grads, scaler.
            loss_scale(), models_are_masters=False)
    if len(fp16_grads_needing_unscale_with_stash) > 0:
        scaler.unscale_with_stashed(fp16_grads_needing_unscale_with_stash,
            preexisting_fp32_grads, preexisting_fp32_grads)
    grads_needing_unscale = []
    grads_needing_unscale_with_stash = []
    stashed = []
    for param, stashed_grad in zip(stash.all_fp32_from_fp32_params, stash.
        all_fp32_from_fp32_grad_stash):
        if param.grad is None and stashed_grad is not None:
            param.grad = stashed_grad
        elif param.grad is not None and stashed_grad is None:
            grads_needing_unscale.append(param.grad)
        elif param.grad is not None and stashed_grad is not None:
            grads_needing_unscale_with_stash.append(param.grad)
            stashed.append(stashed_grad)
        else:
            continue
    if len(grads_needing_unscale) > 0:
        scaler.unscale(grads_needing_unscale, grads_needing_unscale, scaler
            .loss_scale(), models_are_masters=True)
    if len(grads_needing_unscale_with_stash) > 0:
        scaler.unscale_with_stashed(grads_needing_unscale_with_stash,
            stashed, grads_needing_unscale_with_stash)
    for i in range(len(stash.all_fp32_from_fp32_grad_stash)):
        stash.all_fp32_from_fp32_grad_stash[i] = None
