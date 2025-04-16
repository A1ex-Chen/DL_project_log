def __init__(self, **kwargs):
    if not kwargs.get('name', None):
        kwargs['name'] = 'tpu_batch_normalization'
    super(SyncBatchNormalization, self).__init__(**kwargs)
