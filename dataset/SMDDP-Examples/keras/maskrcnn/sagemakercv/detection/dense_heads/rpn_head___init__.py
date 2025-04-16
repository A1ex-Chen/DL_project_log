def __init__(self, rpn_loss_cfg=dict(min_level=2, max_level=6,
    box_loss_type='huber', train_batch_size_per_gpu=1,
    rpn_batch_size_per_im=256, label_smoothing=0.0, rpn_box_loss_weight=1.0
    ), *args, **kwargs):
    super(StandardRPNHead, self).__init__(*args, **kwargs)
    self.loss = RPNLoss(**rpn_loss_cfg)
