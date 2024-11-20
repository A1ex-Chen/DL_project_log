def patch_step(opt, loss_scaler, loss_id):
    opt_step = opt.step

    def skip_step(closure=None):
        if closure is not None:
            raise RuntimeError(
                'Currently, Amp does not support closure use with optimizers.')
        maybe_print(('Gradient overflow.  Skipping step, loss scaler ' +
            '{} reducing loss scale to {}').format(loss_id, loss_scaler.
            loss_scale()))
        if hasattr(opt._amp_stash, 'all_fp32_from_fp16_params'):
            for param in opt._amp_stash.all_fp32_from_fp16_params:
                param.grad = None
        opt.step = opt_step
        opt._amp_stash.already_patched = False
    return skip_step
