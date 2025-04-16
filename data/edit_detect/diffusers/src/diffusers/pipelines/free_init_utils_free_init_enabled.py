@property
def free_init_enabled(self):
    return hasattr(self, '_free_init_num_iters'
        ) and self._free_init_num_iters is not None
