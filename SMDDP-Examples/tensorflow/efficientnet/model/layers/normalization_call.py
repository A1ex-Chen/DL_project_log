def call(self, *args, **kwargs):
    outputs = super(SyncBatchNormalization, self).call(*args, **kwargs)
    return outputs
