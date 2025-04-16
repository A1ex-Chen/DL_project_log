def update_stats(self):
    """
        Update the model with precise statistics. Users can manually call this method.
        """
    if self._disabled:
        return
    if self._data_iter is None:
        self._data_iter = iter(self._data_loader)

    def data_loader():
        for num_iter in itertools.count(1):
            if num_iter % 100 == 0:
                self._logger.info('Running precise-BN ... {}/{} iterations.'
                    .format(num_iter, self._num_iter))
            yield next(self._data_iter)
    with EventStorage():
        self._logger.info('Running precise-BN for {} iterations...  '.
            format(self._num_iter) +
            'Note that this could produce different statistics every time.')
        update_bn_stats(self._model, data_loader(), self._num_iter)
