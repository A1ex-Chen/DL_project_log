@property
def metrics(self):
    return [self.total_loss_tracker, self.reconstruction_loss_tracker, self
        .vq_loss_tracker]
