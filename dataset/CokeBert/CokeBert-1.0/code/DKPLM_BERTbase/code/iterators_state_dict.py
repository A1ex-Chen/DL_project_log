def state_dict(self):
    """Returns a dictionary containing a whole state of the iterator."""
    return {'epoch': self.epoch, 'iterations_in_epoch': self.
        iterations_in_epoch}
