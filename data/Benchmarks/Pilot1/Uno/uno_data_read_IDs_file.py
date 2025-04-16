def read_IDs_file(fname):
    with open(fname, 'r') as f:
        read_ids = f.read().splitlines()
    logger.info('Read file: {}'.format(fname))
    logger.info('Number of elements read: {}'.format(len(read_ids)))
    return read_ids
