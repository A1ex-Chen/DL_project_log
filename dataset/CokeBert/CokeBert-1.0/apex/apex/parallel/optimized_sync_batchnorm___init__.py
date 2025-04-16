def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
    track_running_stats=True, process_group=None, channel_last=False):
    super(SyncBatchNorm, self).__init__(num_features, eps=eps, momentum=
        momentum, affine=affine, track_running_stats=track_running_stats)
    self.process_group = process_group
    self.channel_last = channel_last
