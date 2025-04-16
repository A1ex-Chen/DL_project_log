def accumulate(self):
    """
        Accumulate per image evaluation results and store the result in self.eval.  Does not
        support changing parameter settings from those used by self.evaluate()
        """
    logger.info('Accumulating evaluation results...')
    tic = time.time()
    assert hasattr(self, '_evalImgs_cpp'
        ), 'evaluate() must be called before accmulate() is called.'
    self.eval = _C.COCOevalAccumulate(self._paramsEval, self._evalImgs_cpp)
    self.eval['recall'] = np.array(self.eval['recall']).reshape(self.eval[
        'counts'][:1] + self.eval['counts'][2:])
    self.eval['precision'] = np.array(self.eval['precision']).reshape(self.
        eval['counts'])
    self.eval['scores'] = np.array(self.eval['scores']).reshape(self.eval[
        'counts'])
    toc = time.time()
    logger.info('COCOeval_opt.accumulate() finished in {:0.2f} seconds.'.
        format(toc - tic))
