def post_backward_no_master_weights(self, scaler):
    stash = self._amp_stash
    split_types = (stash.all_fp16_params, stash.all_fp16_grad_stash), (stash
        .all_fp32_params, stash.all_fp32_grad_stash)
    for params, stashed_grads in split_types:
        grads_needing_unscale = []
        grads_needing_unscale_with_stash = []
        stashed = []
        for param, stashed_grad in zip(params, stashed_grads):
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
            scaler.unscale(grads_needing_unscale, grads_needing_unscale,
                scaler.loss_scale(), models_are_masters=True)
        if len(grads_needing_unscale_with_stash) > 0:
            scaler.unscale_with_stashed(grads_needing_unscale_with_stash,
                stashed, grads_needing_unscale_with_stash)
        for i in range(len(stashed_grads)):
            stashed_grads[i] = None
