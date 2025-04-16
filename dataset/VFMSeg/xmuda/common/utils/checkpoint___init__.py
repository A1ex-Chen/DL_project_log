def __init__(self, *args, max_to_keep=5, **kwargs):
    super(CheckpointerV2, self).__init__(*args, **kwargs)
    self.max_to_keep = max_to_keep
    self._last_checkpoints = []
