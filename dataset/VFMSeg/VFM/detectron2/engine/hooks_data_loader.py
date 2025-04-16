def data_loader():
    for num_iter in itertools.count(1):
        if num_iter % 100 == 0:
            self._logger.info('Running precise-BN ... {}/{} iterations.'.
                format(num_iter, self._num_iter))
        yield next(self._data_iter)
