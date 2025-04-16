def _remove_args(self, slices):
    [slice.pop('args', None) for slice in slices]
