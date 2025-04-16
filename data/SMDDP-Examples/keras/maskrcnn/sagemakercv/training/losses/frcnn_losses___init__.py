def __init__(self, mrcnn_weight_loss_mask=1.0, label_smoothing=0.0):
    self.mrcnn_weight_loss_mask = mrcnn_weight_loss_mask
    self.label_smoothing = label_smoothing
