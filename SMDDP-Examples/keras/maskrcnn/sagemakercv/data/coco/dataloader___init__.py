def __init__(self, file_pattern, params, mode=tf.estimator.ModeKeys.TRAIN,
    batch_size=1, num_examples=0, use_fake_data=False, use_instance_mask=
    False, seed=None, disable_options=False, dist_eval=True, data_slack=False):
    self._mode = mode
    self._file_pattern = file_pattern
    self._batch_size = batch_size
    self._num_examples = num_examples
    self._use_fake_data = use_fake_data
    self._use_instance_mask = use_instance_mask
    self._seed = seed
    self._disable_options = disable_options
    self._dist_eval = dist_eval
    self._data_slack = data_slack
    self._params = params
