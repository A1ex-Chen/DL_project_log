def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if self.weight_decay_rate == 0:
        return False
    if self._include_in_weight_decay:
        for r in self._include_in_weight_decay:
            if re.search(r, param_name) is not None:
                return True
    if self._exclude_from_weight_decay:
        for r in self._exclude_from_weight_decay:
            if re.search(r, param_name) is not None:
                return False
    return True
