def clip_grad_norm(p: _GradientClipperInput):
    torch.nn.utils.clip_grad_norm_(p, cfg.CLIP_VALUE, cfg.NORM_TYPE)
