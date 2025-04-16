@property
def metrics(self):
    return [self.total_loss_tracker, self.reconstruction_loss_tracker, self
        .vq_loss_tracker, self.total_pixel_cnn_loss_tracker, self.
        acc_tracker, self.extra_time]
