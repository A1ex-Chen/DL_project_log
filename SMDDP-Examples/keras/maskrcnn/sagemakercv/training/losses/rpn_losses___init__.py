def __init__(self, min_level=2, max_level=6, box_loss_type='huber',
    train_batch_size_per_gpu=1, rpn_batch_size_per_im=256, label_smoothing=
    0.0, rpn_box_loss_weight=1.0):
    self.min_level = min_level
    self.max_level = max_level
    self.box_loss_type = box_loss_type
    self.train_batch_size_per_gpu = train_batch_size_per_gpu
    self.rpn_batch_size_per_im = rpn_batch_size_per_im
    self.label_smoothing = label_smoothing
    self.rpn_box_loss_weight = rpn_box_loss_weight
