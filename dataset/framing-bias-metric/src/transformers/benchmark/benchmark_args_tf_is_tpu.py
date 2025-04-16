@property
@tf_required
def is_tpu(self) ->bool:
    return self._setup_tpu is not None
