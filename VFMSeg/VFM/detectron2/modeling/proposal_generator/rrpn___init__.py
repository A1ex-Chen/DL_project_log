@configurable
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if self.anchor_boundary_thresh >= 0:
        raise NotImplementedError(
            'anchor_boundary_thresh is a legacy option not implemented for RRPN.'
            )
