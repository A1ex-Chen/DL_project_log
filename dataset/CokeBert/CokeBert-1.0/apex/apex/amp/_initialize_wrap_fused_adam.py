def wrap_fused_adam(optimizer, properties):
    msg = (
        'Currently, the usage of FusedAdam is restricted to amp.initialize(..., opt_level="O2", keep_batchnorm_fp32=False, loss_scale=float or "dynamic").  We are working on enabling more general usage.'
        )
    assert properties.master_weights is True, msg
    assert properties.cast_model_type is torch.float16, msg
    assert properties.keep_batchnorm_fp32 is False or properties.keep_batchnorm_fp32 is None, msg
    if properties.loss_scale == 'dynamic':
        return FP16_Optimizer_for_fused(optimizer, dynamic_loss_scale=True)
    else:
        return FP16_Optimizer_for_fused(optimizer, static_loss_scale=
            properties.loss_scale)
