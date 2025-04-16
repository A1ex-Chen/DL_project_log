def clip_grad_value(p: _GradientClipperInput):
    torch.nn.utils.clip_grad_value_(p, cfg.CLIP_VALUE)
