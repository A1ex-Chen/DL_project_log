def state_dict(self):
    return {'base_lrs': self.base_lrs, 'last_epoch': self.last_epoch}
