def state_dict(self):
    ret = super().state_dict()
    ret['_trainer'] = self._trainer.state_dict()
    return ret
