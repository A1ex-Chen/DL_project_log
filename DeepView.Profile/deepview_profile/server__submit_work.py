def _submit_work(self, func, *args, **kwargs):
    print('submit_work', func)
    logger.debug('submit_work args:', args)
    print('submit_work kwargs:', kwargs)
    self._main_executor.submit(func, *args, **kwargs)
