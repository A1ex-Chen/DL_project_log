def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
    track_running_stats=True, process_group=None, channel_last=False):
    if channel_last == True:
        raise AttributeError(
            'channel_last is not supported by primitive SyncBatchNorm implementation. Try install apex with `--cuda_ext` if channel_last is desired.'
            )
    if not SyncBatchNorm.warned:
        if hasattr(self, 'syncbn_import_error'):
            print(
                'Warning:  using Python fallback for SyncBatchNorm, possibly because apex was installed without --cuda_ext.  The exception raised when attempting to import the cuda backend was: '
                , self.syncbn_import_error)
        else:
            print('Warning:  using Python fallback for SyncBatchNorm')
        SyncBatchNorm.warned = True
    super(SyncBatchNorm, self).__init__(num_features, eps=eps, momentum=
        momentum, affine=affine, track_running_stats=track_running_stats)
    self.process_group = process_group
