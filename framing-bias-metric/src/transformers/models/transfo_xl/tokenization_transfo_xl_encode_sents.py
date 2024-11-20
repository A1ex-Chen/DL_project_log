@torch_only_method
def encode_sents(self, sents, ordered=False, verbose=False):
    if verbose:
        logger.info('encoding {} sents ...'.format(len(sents)))
    encoded = []
    for idx, symbols in enumerate(sents):
        if verbose and idx > 0 and idx % 500000 == 0:
            logger.info('    line {}'.format(idx))
        encoded.append(self.convert_to_tensor(symbols))
    if ordered:
        encoded = torch.cat(encoded)
    return encoded
