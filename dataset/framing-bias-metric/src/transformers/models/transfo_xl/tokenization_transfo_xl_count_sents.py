def count_sents(self, sents, verbose=False):
    """
        sents : a list of sentences, each a list of tokenized symbols
        """
    if verbose:
        logger.info('counting {} sents ...'.format(len(sents)))
    for idx, symbols in enumerate(sents):
        if verbose and idx > 0 and idx % 500000 == 0:
            logger.info('    line {}'.format(idx))
        self.counter.update(symbols)
