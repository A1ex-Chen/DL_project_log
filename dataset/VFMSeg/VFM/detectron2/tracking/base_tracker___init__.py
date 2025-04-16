@configurable
def __init__(self, **kwargs):
    self._prev_instances = None
    self._matched_idx = set()
    self._matched_ID = set()
    self._untracked_prev_idx = set()
    self._id_count = 0
