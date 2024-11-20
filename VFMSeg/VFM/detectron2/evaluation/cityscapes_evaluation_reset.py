def reset(self):
    self._working_dir = tempfile.TemporaryDirectory(prefix='cityscapes_eval_')
    self._temp_dir = self._working_dir.name
    assert comm.get_local_size() == comm.get_world_size(
        ), 'CityscapesEvaluator currently do not work with multiple machines.'
    self._temp_dir = comm.all_gather(self._temp_dir)[0]
    if self._temp_dir != self._working_dir.name:
        self._working_dir.cleanup()
    self._logger.info(
        'Writing cityscapes results to temporary directory {} ...'.format(
        self._temp_dir))
