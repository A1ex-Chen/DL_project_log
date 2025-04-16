def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if self.exclude_from_weight_decay:
        for r in self.exclude_from_weight_decay:
            if re.search(r, param_name) is not None:
                return False
    return True
