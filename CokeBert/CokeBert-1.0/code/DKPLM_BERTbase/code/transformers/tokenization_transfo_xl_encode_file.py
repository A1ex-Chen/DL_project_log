def encode_file(self, path, ordered=False, verbose=False, add_eos=True,
    add_double_eos=False):
    if verbose:
        logger.info('encoding file {} ...'.format(path))
    assert os.path.exists(path)
    encoded = []
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if verbose and idx > 0 and idx % 500000 == 0:
                logger.info('    line {}'.format(idx))
            symbols = self.tokenize(line, add_eos=add_eos, add_double_eos=
                add_double_eos)
            encoded.append(self.convert_to_tensor(symbols))
    if ordered:
        encoded = torch.cat(encoded)
    return encoded
