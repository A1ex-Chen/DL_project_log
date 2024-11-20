def count_file(self, path, verbose=False, add_eos=False):
    if verbose:
        logger.info('counting file {} ...'.format(path))
    assert os.path.exists(path), f'Input file {path} not found'
    sents = []
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if verbose and idx > 0 and idx % 500000 == 0:
                logger.info('    line {}'.format(idx))
            symbols = self.tokenize(line, add_eos=add_eos)
            self.counter.update(symbols)
            sents.append(symbols)
    return sents
