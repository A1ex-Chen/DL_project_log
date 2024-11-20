def channels(self, idx=None):
    """feature channels accessor"""
    return self.get('num_chs', idx)
